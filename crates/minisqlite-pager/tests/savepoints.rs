//! Savepoint primitive on the `Pager` seam, exercised through the in-memory
//! backing (`MemPager`), which shares the one copy-on-write layer (`Cow`) with the
//! disk backings — so what holds here holds for every backing.
//!
//! Savepoints are PURE overlay state: `savepoint` / `release_savepoint` /
//! `rollback_to_savepoint` manipulate a bounded pre-image (delta) stack held until
//! the transaction commits, with no journal/WAL interaction. These tests pin the
//! documented behavior (lang_savepoint.html §2/§3):
//!   * `rollback to` restores the EXACT bytes (and page count) as of the mark, and
//!     keeps the savepoint reusable;
//!   * nested savepoints roll back independently;
//!   * `release` keeps the inner changes and merges the delta down so an outer
//!     rollback still restores correctly;
//!   * a savepoint that started the transaction commits on the release of that
//!     outermost savepoint;
//!   * a full `rollback`/`commit` clears the whole savepoint stack.

use minisqlite_pager::{MemPager, PageId, Pager};

const PS: u32 = 512;

fn page_of(byte: u8) -> Vec<u8> {
    vec![byte; PS as usize]
}

/// A committed pager holding `n` pages, page `i` filled with byte `i` (1-based) so
/// each page is distinguishable. Returns with no active transaction.
fn committed(n: PageId) -> MemPager {
    let mut p = MemPager::new(PS);
    p.begin().unwrap();
    for i in 1..=n {
        let id = p.allocate_page().unwrap();
        assert_eq!(id, i);
        p.write_page(id, &page_of(i as u8)).unwrap();
    }
    p.commit().unwrap();
    assert_eq!(p.page_count().unwrap(), n);
    p
}

/// `SAVEPOINT` with no active transaction starts one; `rollback to` restores the
/// exact pre-mark bytes; and the savepoint stays usable for a second write+rollback.
#[test]
fn savepoint_starts_txn_and_rollback_to_restores_exact_bytes() {
    let mut p = committed(2); // page 2 == 0x02 committed

    let sp = p.savepoint().unwrap(); // begins the transaction implicitly
    assert_eq!(sp, 0, "the first savepoint is at depth 0");

    p.write_page(2, &page_of(0xBB)).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xBB)[..], "write visible in the txn");

    p.rollback_to_savepoint(sp).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..], "rolled back to the committed bytes");

    // The savepoint is KEPT and reusable: another write then rollback restores again.
    p.write_page(2, &page_of(0xCC)).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xCC)[..]);
    p.rollback_to_savepoint(sp).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..], "savepoint reusable after rollback to");

    // Releasing the outermost savepoint that started the txn commits it.
    p.release_savepoint(sp).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..]);
    // With the txn committed, a fresh savepoint cycle sees 0x02 as the baseline.
    let sp2 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xDD)).unwrap();
    p.rollback_to_savepoint(sp2).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..], "0x02 was the committed baseline");
    p.release_savepoint(sp2).unwrap();
}

/// Nested savepoints roll back INDEPENDENTLY: `rollback to` the inner mark undoes
/// only work after it; a later `rollback to` the outer mark undoes the rest.
#[test]
fn nested_savepoints_roll_back_independently() {
    let mut p = committed(2); // page 2 == 0x02 committed

    let sp0 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xB0)).unwrap(); // change under sp0

    let sp1 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xC1)).unwrap(); // change under sp1
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xC1)[..]);

    p.rollback_to_savepoint(sp1).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xB0)[..], "only sp1's change undone");

    p.rollback_to_savepoint(sp0).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..], "sp0's change undone too");

    p.release_savepoint(sp0).unwrap(); // commits (sp0 started the txn)
}

/// `rollback to` the OUTER of two savepoints, when the SAME page was written under
/// both while both are still LIVE, must apply the pre-image deltas TOP-DOWN
/// (innermost first, so the outer savepoint's older pre-image wins). This is the one
/// case the nested tests above miss: they roll back the inner savepoint first (which
/// clears its delta), so a page is never recorded at two live levels during a direct
/// rollback to the outer one. A bottom-first apply would wrongly leave the inner
/// savepoint's newer pre-image in the overlay.
#[test]
fn rollback_to_outer_applies_deltas_top_down_for_a_shared_page() {
    let mut p = committed(2); // page 2 == 0x02 committed

    let a = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xA1)).unwrap(); // records a[2] = was-absent (base 0x02)
    let _b = p.savepoint().unwrap(); // inner savepoint; targeted only via `a` below
    p.write_page(2, &page_of(0xB2)).unwrap(); // records _b[2] = 0xA1

    // Both `a` and `b` now hold a live pre-image for page 2. Rolling STRAIGHT back to
    // `a` must restore the pre-`a` bytes (0x02): `a`'s was-absent marker has to win
    // over `b`'s 0xA1, which happens only if the deltas apply innermost (`b`) first
    // and `a` last. A bottom-first apply would remove the entry for `a` then re-insert
    // `b`'s 0xA1, leaving the wrong bytes.
    p.rollback_to_savepoint(a).unwrap();
    assert_eq!(
        p.read_page(2).unwrap(),
        &page_of(0x02)[..],
        "rollback to the outer savepoint restores pre-outer bytes (top-down apply); bottom-first would leave 0xA1",
    );
    p.release_savepoint(a).unwrap();
}

/// `release` of an inner savepoint KEEPS its changes and merges its delta into the
/// enclosing savepoint, so a later `rollback to` the outer mark still restores the
/// pre-outer bytes. Uses an explicit `BEGIN` as the enclosing transaction.
#[test]
fn release_keeps_changes_and_merges_delta_down() {
    let mut p = committed(2); // page 2 == 0x02 committed
    p.begin().unwrap(); // explicit enclosing transaction

    let sp0 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xB0)).unwrap();

    let sp1 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xC1)).unwrap();

    p.release_savepoint(sp1).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xC1)[..], "release keeps the inner change");
    // sp1 is gone: addressing it now is out of range.
    assert!(p.rollback_to_savepoint(sp1).is_err(), "released savepoint no longer exists");

    // The merged delta still lets the outer rollback restore the PRE-sp0 bytes
    // (0x02), proving the merge preserved sp0's older 'was-absent' pre-image rather
    // than adopting sp1's newer one.
    p.rollback_to_savepoint(sp0).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..], "outer rollback restores pre-sp0 state");

    // The enclosing BEGIN still owns the transaction; a full rollback ends it.
    p.rollback().unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..]);
}

/// `rollback to` reverts page-count growth: pages allocated after the mark become
/// out of range again, and the next allocation reuses the same id.
#[test]
fn rollback_to_reverts_page_count_allocation() {
    let mut p = committed(2); // pages 1..=2 committed

    let sp = p.savepoint().unwrap();
    let grown = p.allocate_page().unwrap();
    assert_eq!(grown, 3, "allocation grows to page 3");
    p.write_page(grown, &page_of(0x33)).unwrap();
    assert_eq!(p.page_count().unwrap(), 3);

    p.rollback_to_savepoint(sp).unwrap();
    assert_eq!(p.page_count().unwrap(), 2, "rollback to forgets the grown page");
    assert!(p.read_page(3).is_err(), "the grown page is out of range after rollback to");

    // Growing again reuses id 3 (the count restarts from 2).
    let regrown = p.allocate_page().unwrap();
    assert_eq!(regrown, 3, "the reverted allocation is available again");
    assert_eq!(p.page_count().unwrap(), 3);
    p.release_savepoint(sp).unwrap();
    assert_eq!(p.page_count().unwrap(), 3, "release keeps the re-grown page");
}

/// A savepoint that started the transaction commits on the RELEASE of that outermost
/// savepoint (lang_savepoint §2: "outermost RELEASE == COMMIT"); an ENCLOSING
/// `BEGIN` instead keeps the transaction open across the same release.
#[test]
fn outermost_release_commits_only_when_savepoint_started_the_txn() {
    // Savepoint-started: release(0) commits, so the write survives a fresh txn's
    // rollback (it is the committed baseline now).
    let mut p = committed(2);
    let sp = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xB0)).unwrap();
    p.release_savepoint(sp).unwrap(); // commits
    let sp2 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xEE)).unwrap();
    p.rollback_to_savepoint(sp2).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xB0)[..], "released write became the committed baseline");
    p.rollback().unwrap();
    // Distinguishing assertion: after this FULL rollback drops the second (empty)
    // transaction, `read_page` resolves to the COMMITTED store — the only place the
    // "did the RELEASE commit?" question has a divergent answer. A RELEASE that did
    // NOT commit would leave the store at the pre-0xB0 baseline (0x02); the committed
    // 0xB0 proves the outermost savepoint-started RELEASE was a real, durable commit.
    assert_eq!(
        p.read_page(2).unwrap(),
        &page_of(0xB0)[..],
        "the savepoint-started RELEASE committed 0xB0 to the store, not just the overlay",
    );

    // BEGIN-enclosed: release(0) keeps the txn open, so an outer rollback still
    // discards the released work.
    let mut q = committed(2);
    q.begin().unwrap();
    let qsp = q.savepoint().unwrap();
    q.write_page(2, &page_of(0xB0)).unwrap();
    q.release_savepoint(qsp).unwrap(); // does NOT commit; BEGIN still owns the txn
    assert_eq!(q.read_page(2).unwrap(), &page_of(0xB0)[..], "released change visible within the txn");
    q.rollback().unwrap(); // the enclosing rollback discards it
    assert_eq!(q.read_page(2).unwrap(), &page_of(0x02)[..], "outer rollback undoes the released work");
}

/// A full `rollback` clears the whole savepoint stack; addressing a savepoint after
/// it (or with no active transaction) fails closed rather than corrupting state.
#[test]
fn full_rollback_and_commit_clear_the_savepoint_stack() {
    let mut p = committed(2);
    let sp = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xB0)).unwrap();
    p.rollback().unwrap(); // full rollback ends the txn and drops all savepoints

    assert!(p.release_savepoint(sp).is_err(), "no savepoint after a full rollback");
    assert!(p.rollback_to_savepoint(sp).is_err(), "no savepoint after a full rollback");
    assert_eq!(p.read_page(2).unwrap(), &page_of(0x02)[..], "rollback reverted the write");

    // Same for commit: a committed savepoint-started txn leaves an empty stack.
    let sp2 = p.savepoint().unwrap();
    p.write_page(2, &page_of(0xB1)).unwrap();
    p.commit().unwrap(); // full commit through the Pager seam
    assert!(p.release_savepoint(sp2).is_err(), "no savepoint after a full commit");
    assert_eq!(p.read_page(2).unwrap(), &page_of(0xB1)[..], "commit persisted the write");
}

/// Addressing a non-existent depth fails closed (the engine never does this — it maps
/// names to live depths — but the primitive must not panic or corrupt on a bad index).
#[test]
fn out_of_range_depth_is_rejected() {
    let mut p = committed(1);
    assert!(p.release_savepoint(0).is_err(), "no open savepoint yet");
    assert!(p.rollback_to_savepoint(0).is_err(), "no open savepoint yet");
    let sp = p.savepoint().unwrap();
    assert_eq!(sp, 0);
    assert!(p.rollback_to_savepoint(1).is_err(), "depth 1 is past the only savepoint");
    assert!(p.release_savepoint(5).is_err(), "depth 5 is past the only savepoint");
    // The real savepoint still works after the rejected calls.
    p.write_page(1, &page_of(0x99)).unwrap();
    p.rollback_to_savepoint(0).unwrap();
    assert_eq!(p.read_page(1).unwrap(), &page_of(0x01)[..]);
    p.release_savepoint(0).unwrap();
}
