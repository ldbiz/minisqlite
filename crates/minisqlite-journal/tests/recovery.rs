//! End-to-end recovery tests: drive the real [`Journal`] writer and [`recover`]
//! playback against real files on disk, covering the happy path, the no-op states a
//! clean commit leaves, torn-write truncation, database growth *and* shrink rollback,
//! the multi-segment and -1/EOF journal shapes, the defensive replay stops (a page-0
//! record, a segment with a mismatched page size), every `finish` commit mode, and a
//! byte-exact round-trip of a page that holds a genuine SQLite database header.
//!
//! Temp files live under the OS temp dir with unique names and are removed on drop;
//! nothing is left behind and no fixed path is shared between tests.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use minisqlite_fileformat::DatabaseHeader;
use minisqlite_journal::{
    encode_page_record, page_record_len, recover, Journal, JournalHeader, JournalMode,
    DEFAULT_SECTOR_SIZE, PAGE_COUNT_TO_EOF,
};

/// A database + its `-journal` sibling under the temp dir, removed on drop.
struct Case {
    db: PathBuf,
    journal: PathBuf,
}

impl Case {
    fn new(tag: &str) -> Case {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut db = std::env::temp_dir();
        db.push(format!("msj-it-{tag}-{pid}-{n}-{nanos}.db"));
        let journal = db.with_file_name(format!(
            "{}-journal",
            db.file_name().unwrap().to_string_lossy()
        ));
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
fn page(byte: u8, size: usize) -> Vec<u8> {
    vec![byte; size]
}

/// Write pages concatenated as a database file.
fn write_db(path: &Path, pages: &[Vec<u8>]) {
    let mut f = File::create(path).unwrap();
    for p in pages {
        f.write_all(p).unwrap();
    }
    f.sync_all().unwrap();
}

/// Overwrite a single page in an existing database file (1-based page number).
fn mutate_page(path: &Path, page_no: u32, bytes: &[u8]) {
    let mut f = OpenOptions::new().write(true).open(path).unwrap();
    f.seek(SeekFrom::Start((page_no as u64 - 1) * bytes.len() as u64)).unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}

fn read_file(path: &Path) -> Vec<u8> {
    let mut v = Vec::new();
    File::open(path).unwrap().read_to_end(&mut v).unwrap();
    v
}

/// Write a hot journal the way a crashed transaction would leave one: header,
/// pre-images, synced, but never committed (the file remains on disk).
fn write_hot_journal(case: &Case, page_size: u32, initial_pages: u32, preimages: &[(u32, Vec<u8>)]) {
    let mut j = Journal::create(&case.journal, page_size, initial_pages).unwrap();
    for (no, bytes) in preimages {
        j.append_page(*no, bytes).unwrap();
    }
    j.sync().unwrap();
    // Drop without committing: the journal file survives = a hot journal.
    drop(j);
}

#[test]
fn happy_path_restores_all_pages_and_removes_journal() {
    let case = Case::new("happy");
    let ps = 512usize;
    let originals = vec![page(0xA1, ps), page(0xB2, ps), page(0xC3, ps)];
    write_db(&case.db, &originals);

    let preimages: Vec<(u32, Vec<u8>)> =
        originals.iter().enumerate().map(|(i, p)| (i as u32 + 1, p.clone())).collect();
    write_hot_journal(&case, ps as u32, originals.len() as u32, &preimages);

    // Simulate the half-applied commit: every page is now garbage on disk.
    for i in 1..=originals.len() as u32 {
        mutate_page(&case.db, i, &page(0xFF, ps));
    }

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran, "a hot journal must report recovery ran");

    let restored = read_file(&case.db);
    let expected: Vec<u8> = originals.concat();
    assert_eq!(restored, expected, "every page is restored to its pre-image");
    assert_eq!(restored.len(), originals.len() * ps, "db is exactly the initial size");
    assert!(!case.journal.exists(), "the journal is removed after rollback");
}

#[test]
fn absent_journal_is_noop() {
    let case = Case::new("absent");
    let originals = vec![page(0x10, 512), page(0x20, 512)];
    write_db(&case.db, &originals);

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(!ran, "no journal means nothing to recover");
    assert_eq!(read_file(&case.db), originals.concat(), "db is untouched");
}

#[test]
fn empty_journal_is_noop() {
    let case = Case::new("empty");
    let originals = vec![page(0x10, 512)];
    write_db(&case.db, &originals);
    File::create(&case.journal).unwrap(); // zero-length journal

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(!ran, "an empty journal is not hot");
    assert_eq!(read_file(&case.db), originals.concat(), "db is untouched");
}

#[test]
fn persisted_zeroed_header_is_noop() {
    let case = Case::new("persist");
    let ps = 512usize;
    let originals = vec![page(0x33, ps), page(0x44, ps)];
    write_db(&case.db, &originals);

    // Create a journal, then commit it in PERSIST mode (zeroes the header magic).
    let mut j = Journal::create(&case.journal, ps as u32, originals.len() as u32).unwrap();
    j.append_page(1, &originals[0]).unwrap();
    j.sync().unwrap();
    j.commit_persist().unwrap();
    assert!(case.journal.exists(), "PERSIST leaves the file present");

    // Mutate a page; recovery must NOT roll it back (the transaction committed).
    mutate_page(&case.db, 1, &page(0xEE, ps));
    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(!ran, "a zeroed header is not a hot journal");
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &page(0xEE, ps)[..], "committed change is preserved");
}

#[test]
fn deleted_journal_after_commit_is_noop() {
    let case = Case::new("committed");
    let ps = 512usize;
    let originals = vec![page(0x01, ps)];
    write_db(&case.db, &originals);
    let mut j = Journal::create(&case.journal, ps as u32, 1).unwrap();
    j.append_page(1, &originals[0]).unwrap();
    j.sync().unwrap();
    j.commit_delete().unwrap();
    assert!(!case.journal.exists());

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(!ran, "a deleted journal (a normal commit) leaves nothing to recover");
}

#[test]
fn tiny_malformed_journal_is_noop() {
    let case = Case::new("tiny");
    let originals = vec![page(0x55, 512)];
    write_db(&case.db, &originals);
    // A non-empty file too short to hold a header is not a valid journal.
    std::fs::write(&case.journal, [0xd9, 0xd5, 0x05]).unwrap();

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(!ran, "a header-truncated journal is not hot");
    assert_eq!(read_file(&case.db), originals.concat(), "db is untouched");
}

#[test]
fn valid_magic_but_garbage_page_size_does_not_trash_db() {
    let case = Case::new("garbagehdr");
    let ps = 512usize;
    let originals = vec![page(0x66, ps), page(0x77, ps)];
    write_db(&case.db, &originals);

    // A header with the right magic but an impossible page size must be treated as
    // "not hot" — never as a licence to truncate the database.
    let mut hdr = JournalHeader {
        page_count: 5,
        nonce: 123,
        initial_db_pages: 1, // if trusted, this would truncate the db to one page!
        sector_size: DEFAULT_SECTOR_SIZE,
        page_size: 512,
    }
    .to_padded_header();
    // Corrupt the page-size field (offset 24) to a non-power-of-two.
    hdr[24..28].copy_from_slice(&1000u32.to_be_bytes());
    std::fs::write(&case.journal, &hdr).unwrap();

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(!ran, "a malformed header is not hot");
    assert_eq!(read_file(&case.db), originals.concat(), "db is fully intact, not truncated");
}

#[test]
fn torn_last_record_checksum_replays_prefix_only() {
    let case = Case::new("torn-cksum");
    let ps = 1024usize; // != sector size, to avoid masking an offset bug
    let originals = vec![page(0xA0, ps), page(0xB0, ps), page(0xC0, ps)];
    write_db(&case.db, &originals);

    let preimages: Vec<(u32, Vec<u8>)> =
        originals.iter().enumerate().map(|(i, p)| (i as u32 + 1, p.clone())).collect();
    write_hot_journal(&case, ps as u32, originals.len() as u32, &preimages);

    // Corrupt the checksum of the LAST record directly in the journal file.
    let rec_len = page_record_len(ps);
    let last_record_start = DEFAULT_SECTOR_SIZE as usize + (originals.len() - 1) * rec_len;
    let checksum_off = last_record_start + 4 + ps; // page_no(4) + content(ps)
    {
        let mut f = OpenOptions::new().read(true).write(true).open(&case.journal).unwrap();
        f.seek(SeekFrom::Start(checksum_off as u64)).unwrap();
        let mut b = [0u8; 1];
        f.read_exact(&mut b).unwrap();
        f.seek(SeekFrom::Start(checksum_off as u64)).unwrap();
        f.write_all(&[b[0] ^ 0xFF]).unwrap();
        f.sync_all().unwrap();
    }

    // Mutate all pages; recovery should restore the first two and leave the third.
    for i in 1..=originals.len() as u32 {
        mutate_page(&case.db, i, &page(0xFF, ps));
    }

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran, "a torn journal is still hot; the valid prefix is rolled back");
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &originals[0][..], "page 1 restored");
    assert_eq!(&after[ps..2 * ps], &originals[1][..], "page 2 restored");
    assert_eq!(
        &after[2 * ps..3 * ps],
        &page(0xFF, ps)[..],
        "page 3 (torn record) is NOT restored"
    );
    assert!(!case.journal.exists(), "journal removed after replaying the valid prefix");
}

#[test]
fn truncated_last_record_replays_prefix_only() {
    let case = Case::new("torn-trunc");
    let ps = 512usize;
    let originals = vec![page(0x11, ps), page(0x22, ps), page(0x33, ps)];
    write_db(&case.db, &originals);
    let preimages: Vec<(u32, Vec<u8>)> =
        originals.iter().enumerate().map(|(i, p)| (i as u32 + 1, p.clone())).collect();
    write_hot_journal(&case, ps as u32, originals.len() as u32, &preimages);

    // Cut the journal so the final record is only half-written.
    let rec_len = page_record_len(ps);
    let cut = DEFAULT_SECTOR_SIZE as usize + 2 * rec_len + rec_len / 2;
    OpenOptions::new().write(true).open(&case.journal).unwrap().set_len(cut as u64).unwrap();

    for i in 1..=originals.len() as u32 {
        mutate_page(&case.db, i, &page(0xFF, ps));
    }

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &originals[0][..], "page 1 restored");
    assert_eq!(&after[ps..2 * ps], &originals[1][..], "page 2 restored");
    assert_eq!(&after[2 * ps..3 * ps], &page(0xFF, ps)[..], "half-written page 3 not restored");
}

#[test]
fn rollback_truncates_db_grown_by_aborted_transaction() {
    let case = Case::new("grow");
    let ps = 512usize;
    // The transaction started with 2 pages.
    let originals = vec![page(0x1A, ps), page(0x2B, ps)];
    write_db(&case.db, &originals);

    // Journal the pre-images of the two pages that will change.
    let preimages = vec![(1u32, originals[0].clone()), (2u32, originals[1].clone())];
    write_hot_journal(&case, ps as u32, 2, &preimages);

    // The aborted transaction modified pages 1-2 and grew the file to 4 pages.
    mutate_page(&case.db, 1, &page(0xFF, ps));
    mutate_page(&case.db, 2, &page(0xFF, ps));
    {
        let mut f = OpenOptions::new().write(true).open(&case.db).unwrap();
        f.seek(SeekFrom::Start(2 * ps as u64)).unwrap();
        f.write_all(&page(0xDD, ps)).unwrap(); // page 3
        f.write_all(&page(0xEE, ps)).unwrap(); // page 4
        f.sync_all().unwrap();
    }
    assert_eq!(read_file(&case.db).len(), 4 * ps, "db grew to 4 pages");

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(after.len(), 2 * ps, "db truncated back to its initial 2 pages");
    assert_eq!(after, originals.concat(), "surviving pages restored to pre-images");
}

#[test]
fn multi_segment_journal_replays_every_segment() {
    let case = Case::new("multiseg");
    let ps = 512usize;
    let sector = DEFAULT_SECTOR_SIZE as usize;
    let originals = vec![page(0x0A, ps), page(0x0B, ps), page(0x0C, ps)];
    write_db(&case.db, &originals);

    // Build a two-segment journal by hand: each segment has its OWN nonce and one
    // record, and headers are separated at the next sector boundary.
    let nonce_a = 0x1111_1111u32;
    let nonce_b = 0x2222_2222u32;
    let seg0_hdr = JournalHeader {
        page_count: 1,
        nonce: nonce_a,
        initial_db_pages: 3,
        sector_size: sector as u32,
        page_size: ps as u32,
    };
    let seg1_hdr = JournalHeader { nonce: nonce_b, ..seg0_hdr };

    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&seg0_hdr.to_padded_header());
    bytes.extend_from_slice(&encode_page_record(1, &originals[0], nonce_a));
    // Pad to the next sector boundary before the second header.
    let seg1_start = bytes.len().div_ceil(sector) * sector;
    bytes.resize(seg1_start, 0);
    bytes.extend_from_slice(&seg1_hdr.to_padded_header());
    bytes.extend_from_slice(&encode_page_record(2, &originals[1], nonce_b));
    std::fs::write(&case.journal, &bytes).unwrap();

    // Pages 1 and 2 are garbage; page 3 was never in the journal.
    mutate_page(&case.db, 1, &page(0xFF, ps));
    mutate_page(&case.db, 2, &page(0xFF, ps));

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &originals[0][..], "segment 0 page restored (nonce A)");
    assert_eq!(&after[ps..2 * ps], &originals[1][..], "segment 1 page restored (nonce B)");
    assert_eq!(&after[2 * ps..3 * ps], &originals[2][..], "untouched page unchanged");
}

#[test]
fn eof_sentinel_reads_all_records_and_ignores_partial_tail() {
    let case = Case::new("eof");
    let ps = 512usize;
    let sector = DEFAULT_SECTOR_SIZE as usize;
    let originals = vec![page(0x71, ps), page(0x82, ps)];
    write_db(&case.db, &originals);

    let nonce = 0x9ABC_DEF0u32;
    let hdr = JournalHeader {
        page_count: PAGE_COUNT_TO_EOF, // -1: records run to end of file
        nonce,
        initial_db_pages: 2,
        sector_size: sector as u32,
        page_size: ps as u32,
    };
    let mut bytes = hdr.to_padded_header();
    bytes.extend_from_slice(&encode_page_record(1, &originals[0], nonce));
    bytes.extend_from_slice(&encode_page_record(2, &originals[1], nonce));
    // A trailing partial record (fewer than a full record) must be ignored, not
    // misread as data.
    bytes.resize(bytes.len() + 37, 0);
    std::fs::write(&case.journal, &bytes).unwrap();

    mutate_page(&case.db, 1, &page(0xFF, ps));
    mutate_page(&case.db, 2, &page(0xFF, ps));

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &originals[0][..]);
    assert_eq!(&after[ps..2 * ps], &originals[1][..]);
}

#[test]
fn roundtrips_a_page_holding_a_real_database_header() {
    // Uses a genuine SQLite database header (via minisqlite-fileformat) as page 1 so
    // recovery is exercised against a realistic page, and we confirm the restored
    // bytes still parse as that header.
    let case = Case::new("dbhdr");
    let ps = 512u32;
    let mut header = DatabaseHeader::default();
    header.page_size = ps;
    header.database_size_pages = 2;

    let mut page1 = vec![0u8; ps as usize];
    page1[0..100].copy_from_slice(&header.to_bytes());
    let page2 = page(0x5C, ps as usize);
    let originals = vec![page1.clone(), page2.clone()];
    write_db(&case.db, &originals);

    let preimages = vec![(1u32, page1.clone()), (2u32, page2.clone())];
    write_hot_journal(&case, ps, 2, &preimages);

    mutate_page(&case.db, 1, &page(0x00, ps as usize));
    mutate_page(&case.db, 2, &page(0x00, ps as usize));

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps as usize], &page1[..], "header page restored byte-exact");

    let mut hdr_bytes = [0u8; 100];
    hdr_bytes.copy_from_slice(&after[0..100]);
    let parsed = DatabaseHeader::read(&hdr_bytes).unwrap();
    assert_eq!(parsed, header, "restored header still parses to the original");
}

#[test]
fn unsynced_zero_count_header_still_replays_records() {
    // A journal whose header still reads page_count == 0 (records written but the
    // count never backfilled, i.e. a crash before sync) must still be rolled back by
    // reading records to EOF, matching SQLite's nRec==0 handling.
    let case = Case::new("unsynced");
    let ps = 512usize;
    let originals = vec![page(0x4D, ps), page(0x5E, ps)];
    write_db(&case.db, &originals);

    // Append pre-images but deliberately DO NOT call sync(), so the on-disk header
    // keeps its page_count == 0 placeholder.
    let mut j = Journal::create(&case.journal, ps as u32, originals.len() as u32).unwrap();
    j.append_page(1, &originals[0]).unwrap();
    j.append_page(2, &originals[1]).unwrap();
    drop(j);

    // Sanity: the header count is really 0 on disk.
    let raw = read_file(&case.journal);
    assert_eq!(&raw[8..12], &0u32.to_be_bytes(), "page_count left at 0");

    for i in 1..=originals.len() as u32 {
        mutate_page(&case.db, i, &page(0xFF, ps));
    }
    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    assert_eq!(read_file(&case.db), originals.concat(), "records replayed despite count 0");
}

#[test]
fn finish_dispatches_each_mode_to_its_commit() {
    // Exercise all three arms of `Journal::finish` so a wrong-arm dispatch bug (e.g.
    // Persist routed to commit_delete) cannot slip through.
    let ps = 512u32;

    // Delete arm: the journal file is removed.
    let del = Case::new("finish-del");
    write_db(&del.db, &[page(0x01, ps as usize)]);
    let mut j = Journal::create(&del.journal, ps, 1).unwrap();
    j.append_page(1, &page(0x01, ps as usize)).unwrap();
    j.sync().unwrap();
    j.finish(JournalMode::Delete).unwrap();
    assert!(!del.journal.exists(), "finish(Delete) removes the journal");

    // Truncate arm: the file stays but is zero length (not hot).
    let trunc = Case::new("finish-trunc");
    write_db(&trunc.db, &[page(0x01, ps as usize)]);
    let mut j = Journal::create(&trunc.journal, ps, 1).unwrap();
    j.append_page(1, &page(0x01, ps as usize)).unwrap();
    j.sync().unwrap();
    j.finish(JournalMode::Truncate).unwrap();
    assert!(trunc.journal.exists());
    assert_eq!(read_file(&trunc.journal).len(), 0, "finish(Truncate) zeroes the file");
    assert!(!recover(&trunc.db, &trunc.journal).unwrap(), "truncated journal is not hot");

    // Persist arm: the file stays with a zeroed header magic (not hot).
    let persist = Case::new("finish-persist");
    write_db(&persist.db, &[page(0x01, ps as usize)]);
    let mut j = Journal::create(&persist.journal, ps, 1).unwrap();
    j.append_page(1, &page(0x01, ps as usize)).unwrap();
    j.sync().unwrap();
    j.finish(JournalMode::Persist).unwrap();
    assert!(persist.journal.exists(), "finish(Persist) keeps the file");
    assert_eq!(&read_file(&persist.journal)[0..8], &[0u8; 8], "finish(Persist) zeroes the magic");
    assert!(!recover(&persist.db, &persist.journal).unwrap(), "persisted journal is not hot");
}

#[test]
fn rollback_restores_pages_freed_by_a_shrinking_transaction() {
    // A transaction that shrank the db (freed pages and truncated the file) must be
    // rolled back by RE-EXTENDING the file: recovery writes the freed pages' pre-images
    // back beyond the current EOF, then set_len restores the initial size. This pins
    // the extend-during-restore direction (the growth test only covers truncation).
    let case = Case::new("shrink");
    let ps = 512usize;
    let originals = vec![page(0x1A, ps), page(0x2B, ps), page(0x3C, ps), page(0x4D, ps)];
    write_db(&case.db, &originals);

    // The txn journals the pre-images of the two pages it will free, then truncates.
    let preimages = vec![(3u32, originals[2].clone()), (4u32, originals[3].clone())];
    write_hot_journal(&case, ps as u32, 4, &preimages);

    // The aborted txn truncated the file down to 2 pages before crashing.
    OpenOptions::new().write(true).open(&case.db).unwrap().set_len(2 * ps as u64).unwrap();
    assert_eq!(read_file(&case.db).len(), 2 * ps, "db shrank to 2 pages");

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(after.len(), 4 * ps, "db re-extended to its initial 4 pages");
    assert_eq!(after, originals.concat(), "freed pages restored to their pre-images");
}

#[test]
fn record_with_page_no_zero_stops_replay() {
    // A checksum-valid record naming page 0 is impossible in a real journal; replay
    // must treat it as the end of trustworthy data and stop (recover.rs page_no == 0
    // guard), applying only the records before it.
    let case = Case::new("page0");
    let ps = 512usize;
    let originals = vec![page(0x61, ps), page(0x62, ps)];
    write_db(&case.db, &originals);

    let nonce = 0x1357_9BDFu32;
    let hdr = JournalHeader {
        page_count: 2,
        nonce,
        initial_db_pages: 2,
        sector_size: DEFAULT_SECTOR_SIZE,
        page_size: ps as u32,
    };
    let mut bytes = hdr.to_padded_header();
    bytes.extend_from_slice(&encode_page_record(1, &originals[0], nonce));
    bytes.extend_from_slice(&encode_page_record(0, &page(0x99, ps), nonce)); // page 0!
    std::fs::write(&case.journal, &bytes).unwrap();

    mutate_page(&case.db, 1, &page(0xFF, ps));
    mutate_page(&case.db, 2, &page(0xFF, ps));

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &originals[0][..], "page 1 (before the bad record) restored");
    assert_eq!(&after[ps..2 * ps], &page(0xFF, ps)[..], "replay stopped at the page-0 record");
}

#[test]
fn later_segment_with_mismatched_page_size_stops_replay() {
    // A second segment that declares a different (but individually valid) page size is
    // internally inconsistent with the first, so replay must stop rather than
    // reinterpret its bytes (recover.rs page_size/sector_size mismatch break).
    let case = Case::new("segmismatch");
    let ps = 512usize;
    let sector = DEFAULT_SECTOR_SIZE as usize;
    let originals = vec![page(0x71, ps), page(0x72, ps)];
    write_db(&case.db, &originals);

    let nonce_a = 0xAAAA_0001u32;
    let seg0 = JournalHeader {
        page_count: 1,
        nonce: nonce_a,
        initial_db_pages: 2,
        sector_size: sector as u32,
        page_size: ps as u32,
    };
    // Segment 2 declares page_size 1024 (valid on its own, but != segment 1's 512).
    let seg1 = JournalHeader { page_size: 1024, ..seg0 };

    let mut bytes = seg0.to_padded_header();
    bytes.extend_from_slice(&encode_page_record(1, &originals[0], nonce_a));
    let seg1_start = bytes.len().div_ceil(sector) * sector;
    bytes.resize(seg1_start, 0);
    bytes.extend_from_slice(&seg1.to_padded_header());
    // A record for page 2 in the mismatched segment — must NOT be applied.
    bytes.extend_from_slice(&encode_page_record(2, &page(0x88, ps), nonce_a));
    std::fs::write(&case.journal, &bytes).unwrap();

    mutate_page(&case.db, 1, &page(0xFF, ps));
    mutate_page(&case.db, 2, &page(0xFF, ps));

    let ran = recover(&case.db, &case.journal).unwrap();
    assert!(ran);
    let after = read_file(&case.db);
    assert_eq!(&after[0..ps], &originals[0][..], "segment 0 page 1 restored");
    assert_eq!(&after[ps..2 * ps], &page(0xFF, ps)[..], "mismatched segment 1 skipped");
}
