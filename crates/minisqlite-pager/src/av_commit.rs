//! Commit-time auto_vacuum finalization + FULL compaction, run as ONE durable
//! transaction. [`Cow::commit`](crate::cow::Cow) calls [`finalize_and_compact`] instead of
//! [`crate::ptrmap_build::finalize`] directly, so a FULL auto_vacuum database is COMPACTED
//! inside the very overlay — and the single `apply_commit` — that carries the user's data
//! write.
//!
//! ## Why atomic-in-commit (the defect this fixes)
//! The previous design compacted a FULL database in a SEPARATE, second transaction AFTER
//! the user's commit (an engine-level post-commit sweep). That produced TWO durable commits
//! per free-page-releasing FULL commit, which diverged from real sqlite two ways:
//! - A crash BETWEEN the two commits left a valid, committed, but UN-compacted file
//!   (`freelist_count > 0`) — a state real sqlite never presents for a committed FULL
//!   database, and one nothing re-compacts until the next write. Real sqlite compacts +
//!   truncates a FULL database WITHIN the user's commit, so a committed FULL database always
//!   has `freelist_count == 0`.
//! - Two fsyncs where one suffices.
//!
//! Folding reclaim into the commit closes the crash window (a crash now recovers to either
//! the pre-commit image or the compacted post-commit image, never the un-compacted middle)
//! and costs one fsync.
//!
//! ## Why the order is finalize → reclaim → finalize
//! The operation sequence is exactly the (already proven) two-phase path with the middle
//! durability barrier removed, so the committed image is identical to what that path
//! produced — only the intermediate fsync and its crash window are gone:
//! - `finalize` #1 rebuilds the pointer map and enforces §1.8 roots-first for the forest the
//!   user's writes produced, so [`reclaim`](crate::reclaim) reads a ptrmap that MATCHES the
//!   current forest (its relocation resolves back-pointers from it). After it, roots occupy
//!   the lowest data slots and every free page sits above them.
//! - `reclaim(None)` relocates the trailing live pages into the freed slots and truncates
//!   (the FULL budget). Because every free slot is above every root, it only ever moves
//!   NON-root tail pages down, so roots-first is preserved.
//! - `finalize` #2 re-derives the ptrmap for the now-smaller file — the same
//!   reclaim-then-finalize the standalone `incremental_vacuum` path already performs, and a
//!   roots_first no-op here since reclaim kept roots first.
//!
//! ## Decline is safe; corruption is not
//! `reclaim` (and the re-derive) run inside a bounded overlay UNDO scope
//! ([`Cow::with_reclaim_scope`]). If reclaim's verify-or-abort DECLINES, or any step faults,
//! the scope rolls back to the post-finalize-#1 image and the user's data commits
//! UN-compacted — exactly the state the old path committed in its first phase when its
//! second phase declined. The pre-compaction image is never durably written, so a crash can
//! never expose it; a torn/incorrect file is never written at all.

use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};
use minisqlite_types::Result;

use crate::cow::Cow;
use crate::store::PageStore;

/// Finalize the committing transaction's auto_vacuum structures and, for a FULL auto_vacuum
/// database with reclaimable free pages, COMPACT the file in the same transaction. Returns
/// `true` iff a b-tree ROOT relocated (so [`Cow::commit`] records it for the engine's
/// cached-catalog reload), combining a root move from `finalize` or from the in-commit
/// reclaim / its re-derive.
///
/// Beyond the `finalize` every commit already ran, this adds only one page-1 header read for
/// any database that is not FULL auto_vacuum with free pages — so NONE stays byte-identical,
/// and INCREMENTAL is unchanged (its on-demand `PRAGMA incremental_vacuum` still reclaims via
/// its own transaction).
pub(crate) fn finalize_and_compact<S: PageStore>(cow: &mut Cow<S>) -> Result<bool> {
    // 1. Make the ptrmap valid for the forest the user's writes produced (and enforce §1.8
    //    roots-first). This is the finalize `Cow::commit` ran on its own before compaction
    //    became atomic; `reclaim` below relies on the ptrmap it leaves.
    let mut moved_roots = crate::ptrmap_build::finalize(cow)?;

    // 2. Gate on FULL auto_vacuum (off52 != 0 AND off64 == 0) with something to reclaim.
    //    Read AFTER finalize so `freelist_count` reflects any DROP-leaked pages it swept.
    if !full_with_reclaimable_free_pages(cow)? {
        return Ok(moved_roots);
    }

    // 3. Compact within THIS transaction, under a bounded undo scope so a declined or
    //    faulted relocation rolls the compaction back and the user's data still commits
    //    un-compacted (verify-or-abort: declining is safe, corrupting is not).
    let kept = cow.with_reclaim_scope(|c| {
        let outcome = crate::reclaim::reclaim(c, None)?;
        // Re-derive the ptrmap for the now-smaller file. Inside the scope so a re-derive
        // fault also declines to the post-finalize-#1 image rather than committing a torn
        // file. When reclaim removed nothing the image is already `finalize`'s output, so
        // there is nothing to re-derive.
        //
        // A fault HERE is a "should never happen": finalize #1 already produced a
        // finalize-accepted image and `reclaim`'s `verify_forest` just passed, so re-deriving
        // the strictly smaller, still-consistent forest cannot legitimately fail. If a latent
        // relocation/derive bug ever made it fail, `with_reclaim_scope` maps it to the SAME
        // safe rollback as a genuine verify-or-abort DECLINE — the user's data commits
        // un-compacted rather than corrupt — so the only visible symptom is "this FULL db
        // never compacts", never a torn file. That swallow is the deliberate
        // decline-over-corrupt trade, not an ordinary expected decline.
        let refinalized =
            if outcome.reclaimed > 0 { crate::ptrmap_build::finalize(c)? } else { false };
        Ok((outcome.root_moved, refinalized))
    })?;

    if let Some((reclaim_moved_root, refinalized)) = kept {
        moved_roots = moved_roots || reclaim_moved_root || refinalized;
    }
    Ok(moved_roots)
}

/// Whether the committing database is FULL auto_vacuum — offset 52 largest-root != 0 (auto_
/// vacuum on) AND offset 64 incremental-flag == 0 (FULL, not INCREMENTAL) — and currently
/// holds at least one free page (offset 36). Reads page 1 through the transaction overlay, so
/// it sees the header `finalize` just wrote, including the freelist count after any DROP-leak
/// sweep. A malformed or too-short page 1 — never the real engine path, since `finalize`
/// already validated it — reports `false` (fail closed: skip compaction rather than act on a
/// header we cannot trust).
fn full_with_reclaimable_free_pages<S: PageStore>(cow: &Cow<S>) -> Result<bool> {
    if cow.page_count()? < 1 {
        return Ok(false);
    }
    let page1 = cow.read_page(1)?;
    let Some(head) = page1.get(..HEADER_SIZE) else {
        return Ok(false);
    };
    let mut buf = [0u8; HEADER_SIZE];
    buf.copy_from_slice(head);
    let header = match DatabaseHeader::read(&buf) {
        Ok(h) => h,
        Err(_) => return Ok(false),
    };
    Ok(header.largest_root_btree != 0
        && header.incremental_vacuum == 0
        && header.freelist_count != 0)
}

#[cfg(test)]
mod tests {
    use crate::cow::Cow;
    use crate::store::MemStore;
    use minisqlite_types::Error;

    const PS: u32 = 512;

    fn page(byte: u8) -> Vec<u8> {
        vec![byte; PS as usize]
    }

    /// The verify-or-abort DECLINE machinery at the mechanism level (invariant #1's safety
    /// valve): a DECLINE inside [`Cow::with_reclaim_scope`] — what `reclaim`/`verify_forest`
    /// aborting maps to — must roll the scope's overlay AND page_count back EXACTLY, so the
    /// user's data commits UN-compacted instead of a torn/shrunk file; and a SUCCEEDING scope
    /// must keep its changes (the compact path).
    ///
    /// This drives the scope directly rather than forcing a real reclaim decline, because a
    /// decline is unreachable for a CLEAN forest (finalize #1 builds the matching ptrmap
    /// immediately before reclaim reads it) — `reclaim.rs`'s own unit tests already exercise
    /// `verify_forest`'s decline in isolation. The closure here stages a write and a
    /// savepoint-aware `truncate_to` (the exact mutations a compaction makes) then returns
    /// `Err`, so it proves `rollback_reclaim_scope` + `truncate_to`'s RESTORE side undo them
    /// byte-for-byte — the branch no end-to-end compaction test can reach without a fault.
    #[test]
    fn declined_reclaim_scope_restores_overlay_and_count_exactly() {
        let mut cow = Cow::new(MemStore::new(PS));
        cow.begin().unwrap();
        // Four distinct staged pages: the post-finalize-#1 image a compaction would act on.
        for b in 0..4u8 {
            let id = cow.grow_one().unwrap();
            cow.stage_write(id, &page(0xA0 + b)).unwrap();
        }
        let count_before = cow.page_count().unwrap();
        let snapshot: Vec<Vec<u8>> =
            (1..=count_before).map(|id| cow.read_page(id).unwrap().to_vec()).collect();

        // DECLINE: modify a SURVIVING page and truncate away the tail, then abort. Both
        // mutations capture pre-images into the scope; the abort must restore them all.
        let declined = cow
            .with_reclaim_scope(|c| {
                c.stage_write(2, &page(0xEE))?;
                c.truncate_to(2)?; // drops pages 3,4 (captures their pre-images)
                Err::<(), Error>(Error::Io("forced decline".into()))
            })
            .unwrap();
        assert!(declined.is_none(), "an Err closure is reported as a DECLINE (None)");
        assert_eq!(
            cow.page_count().unwrap(),
            count_before,
            "a declined scope restores the page count (undoes the truncate)"
        );
        for id in 1..=count_before {
            assert_eq!(
                cow.read_page(id).unwrap(),
                &snapshot[(id - 1) as usize][..],
                "a declined scope restores page {id} to its exact pre-scope bytes"
            );
        }

        // KEEP: the same truncate under a SUCCEEDING scope is applied (the compact path).
        let kept = cow.with_reclaim_scope(|c| {
            c.truncate_to(2)?;
            Ok(7u32)
        });
        assert_eq!(kept.unwrap(), Some(7), "a succeeding scope keeps its changes and returns the value");
        assert_eq!(cow.page_count().unwrap(), 2, "a kept scope applied the truncate");
        assert!(cow.read_page(3).is_err(), "the dropped page is out of range after a kept truncate");

        cow.rollback().unwrap();
    }
}
