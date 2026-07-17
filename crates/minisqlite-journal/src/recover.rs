//! Hot-journal playback: the recovery a disk pager runs when it opens a database and
//! finds a leftover `-journal` (atomiccommit §4). If the journal is *hot* (present,
//! non-empty, well-formed header), a previous process crashed mid-commit and the
//! database may be inconsistent; we roll it back by writing each journaled page's
//! original content back into the database, truncating the database to its
//! pre-transaction size, and removing the journal.
//!
//! Recovery is *idempotent*: re-running it after an interruption (say, a crash after
//! syncing the database but before deleting the journal) writes the same pre-images
//! and truncates to the same size, so a partially completed recovery simply completes
//! on the next open. That is why the pre-images are written before the journal is
//! removed, and why we never need to record recovery's own progress.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use minisqlite_types::Result;

use crate::codec::{
    decode_page_record, has_valid_magic, page_record_len, JournalHeader, HEADER_PREFIX_LEN,
};
use crate::util::{remove_file_durable, round_up};

/// Play back a hot rollback journal, if one exists, restoring `db_path` to its
/// pre-transaction state.
///
/// Returns `Ok(true)` if a hot journal was found and rolled back (the database was
/// restored and the journal removed), or `Ok(false)` if there was nothing to
/// recover: the journal is absent, empty, or has an invalid/zeroed header (the states
/// a clean commit — DELETE/TRUNCATE/PERSIST — leaves behind).
///
/// A record whose checksum does not verify marks the end of trustworthy data (a torn
/// final write while the journal itself was still being written, before the database
/// was ever touched): replay stops there, having applied the valid prefix, and is not
/// an error. Only genuine I/O failures on the database are returned as errors.
pub fn recover(db_path: &Path, journal_path: &Path) -> Result<bool> {
    // 1. A journal must exist and be non-empty to be hot.
    let mut journal = match OpenOptions::new().read(true).open(journal_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    let journal_len = journal.metadata()?.len();
    if journal_len == 0 {
        return Ok(false);
    }

    // 2. Parse and validate the opening header. A short, zeroed, or otherwise
    //    malformed header means the journal is not hot — never an error, and in
    //    particular we never trash the database on a garbage header. A file too short
    //    to hold a header is "not hot" (EOF), but a genuine read failure is propagated
    //    rather than silently reported as "nothing to recover".
    let mut prefix = [0u8; HEADER_PREFIX_LEN];
    match read_at(&mut journal, 0, &mut prefix) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(e.into()),
    }
    if !has_valid_magic(&prefix) {
        return Ok(false);
    }
    let first = match JournalHeader::decode(&prefix) {
        Ok(h) => h,
        Err(_) => return Ok(false),
    };
    let page_size = first.page_size as usize;
    let record_len = page_record_len(page_size) as u64;

    // 3. Open the database read-write. If a hot journal names a database that cannot
    //    be opened, that is a real I/O error (fail closed) — not a silent no-op.
    let mut db = OpenOptions::new().read(true).write(true).open(db_path)?;

    // 4. Replay every segment's records until a segment header is missing/invalid
    //    (end of journal) or a record fails to verify (torn write).
    let mut record_buf = vec![0u8; page_record_len(page_size)];
    let mut segment_start: u64 = 0;
    'segments: loop {
        // Read this segment's header. The first segment is guaranteed valid (checked
        // above); a later one that is missing or invalid marks the end of the journal.
        if segment_start + HEADER_PREFIX_LEN as u64 > journal_len {
            break;
        }
        let mut seg_prefix = [0u8; HEADER_PREFIX_LEN];
        match read_at(&mut journal, segment_start, &mut seg_prefix) {
            Ok(()) => {}
            // A short read here is the end of the journal, not a failure.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        if !has_valid_magic(&seg_prefix) {
            break;
        }
        let segment = match JournalHeader::decode(&seg_prefix) {
            Ok(h) => h,
            Err(_) => break,
        };
        // All headers in one journal share page and sector size; a mismatch means we
        // have run off the real records into unrelated bytes, so stop.
        if segment.page_size != first.page_size || segment.sector_size != first.sector_size {
            break;
        }

        let sector = segment.sector_size as u64;
        let nonce = segment.nonce; // each segment carries its own checksum nonce
        let records_start = segment_start + sector;

        // How many records this segment claims. Three cases:
        //  - the -1 sentinel: as many records as fit before end of file;
        //  - a count of 0 in the *first* header: the records were written but the
        //    count was never backfilled (a crash before the journal was synced), so
        //    read to end of file just as SQLite does — the checksums bound it;
        //  - otherwise: the explicit backfilled count.
        // The first two both run to EOF and are the last (or only) segment.
        let read_to_eof = segment.records_to_eof() || (segment.page_count == 0 && segment_start == 0);
        let claimed = if read_to_eof {
            journal_len.saturating_sub(records_start) / record_len
        } else {
            segment.page_count as u64
        };

        let mut record_pos = records_start;
        let mut applied = 0u64;
        while applied < claimed {
            // A record that would run past EOF is a torn final write: stop.
            if record_pos + record_len > journal_len {
                break 'segments;
            }
            match read_at(&mut journal, record_pos, &mut record_buf) {
                Ok(()) => {}
                // A short read is a torn/truncated final record: stop replay.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break 'segments,
                Err(e) => return Err(e.into()),
            }
            let record = decode_page_record(&record_buf, page_size, nonce)?;
            // A bad checksum or an impossible page number ends the trustworthy data.
            if !record.checksum_ok || record.page_no == 0 {
                break 'segments;
            }
            // Restore the original page content to its slot in the database file.
            let offset = (record.page_no as u64 - 1) * page_size as u64;
            db.seek(SeekFrom::Start(offset))?;
            db.write_all(record.content)?;
            record_pos += record_len;
            applied += 1;
        }

        // A read-to-EOF segment is the last one by definition. An explicit zero count
        // (in a later segment) carries no records and no successor, so stop there too.
        if read_to_eof || claimed == 0 {
            break;
        }
        // The next header begins on the next sector boundary after this segment's
        // records. Guard against a non-advancing position so a malformed journal
        // cannot spin forever.
        let next = round_up(record_pos, sector);
        if next <= segment_start || next >= journal_len {
            break;
        }
        segment_start = next;
    }

    // 5. Truncate the database back to its pre-transaction size, undoing any growth
    //    the aborted transaction caused. The header's initial size is trusted once
    //    the header is well-formed (it lives in its own sector, written and synced
    //    before any database change), matching SQLite's rollback. A pre-image can only
    //    exist for a page that existed at transaction start, so `page_no <=
    //    initial_db_pages` always holds for a valid journal; were a corrupt record to
    //    carry a larger page_no (and somehow pass its checksum), this truncation simply
    //    discards that spurious page rather than corrupting the result.
    let target = first.initial_db_pages as u64 * page_size as u64;
    db.set_len(target)?;
    // 6. Flush the restored database before removing the journal, so a crash here
    //    still leaves the journal to redo the (idempotent) rollback on the next open.
    db.sync_all()?;
    drop(db);

    // 7. Remove the journal durably (tolerating an already-gone file and fsync'ing the
    //    parent dir). Its absence is what marks the rollback complete.
    remove_file_durable(journal_path)?;
    Ok(true)
}

/// Read exactly `buf.len()` bytes at absolute offset `off`. A short read surfaces as
/// `ErrorKind::UnexpectedEof`, which callers classify as "end of journal / stop
/// replay" (never a partially filled buffer mistaken for real data); any other I/O
/// error is a genuine failure the callers propagate.
fn read_at(file: &mut File, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(off))?;
    file.read_exact(buf)
}
