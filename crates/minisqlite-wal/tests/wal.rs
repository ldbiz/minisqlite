//! End-to-end tests of the WAL codec + index through the public API: frame
//! round-trip and validation, salt invalidation, mid-frame corruption, the reader
//! `FindFrame` algorithm, snapshot isolation at a recorded mxFrame, and a
//! reference-model sweep of `resolve` against a linear scan.

use minisqlite_wal::{scan, FrameHeader, WalBuilder, WalHeader};

const PS: u32 = 512;

fn page(byte: u8) -> Vec<u8> {
    vec![byte; PS as usize]
}

/// A WAL with a valid header and two committed transactions. Frames:
///   1: page1 (non-commit)      \ txn A
///   2: page2 (commit, db=2)    /
///   3: page1 (non-commit)      \ txn B
///   4: page3 (commit, db=3)    /
fn two_txn_wal() -> Vec<u8> {
    let mut b = WalBuilder::new(WalHeader::new(PS, 0xaa, 0xbb, 0, false));
    b.append_frame(1, 0, &page(0x11)).unwrap();
    b.append_frame(2, 2, &page(0x22)).unwrap();
    b.append_frame(1, 0, &page(0x33)).unwrap();
    b.append_frame(3, 3, &page(0x44)).unwrap();
    b.into_bytes()
}

#[test]
fn scan_validates_all_frames_and_finds_mx() {
    let wal = two_txn_wal();
    let idx = scan(&wal);
    assert!(idx.has_valid_header());
    assert_eq!(idx.page_size(), PS);
    assert_eq!(idx.n_valid_frames(), 4);
    assert_eq!(idx.mx_frame(), 4);
    assert_eq!(idx.db_size_pages(), 3);
    assert!(!idx.is_empty());
}

#[test]
fn frame_page_data_reads_the_right_bytes() {
    let wal = two_txn_wal();
    let idx = scan(&wal);
    assert_eq!(idx.frame_page_data(&wal, 1).unwrap(), &page(0x11)[..]);
    assert_eq!(idx.frame_page_data(&wal, 4).unwrap(), &page(0x44)[..]);
    assert!(idx.frame_page_data(&wal, 0).is_err());
    assert!(idx.frame_page_data(&wal, 5).is_err());
}

#[test]
fn reader_algorithm_resolves_latest_committed() {
    let wal = two_txn_wal();
    let idx = scan(&wal);
    let mx = idx.mx_frame();
    // Latest committed instance of each page at the full snapshot.
    assert_eq!(idx.resolve(1, mx), Some(3)); // page1 newest at frame 3
    assert_eq!(idx.resolve(2, mx), Some(2));
    assert_eq!(idx.resolve(3, mx), Some(4));
    // A page never written to the WAL ⇒ read from the db file.
    assert_eq!(idx.resolve(99, mx), None);
    // And the resolved page data matches.
    assert_eq!(idx.resolve_page_data(&wal, 1, mx).unwrap(), Some(&page(0x33)[..]));
    assert_eq!(idx.resolve_page_data(&wal, 99, mx).unwrap(), None);
}

#[test]
fn snapshot_at_earlier_commit_sees_only_first_txn() {
    let wal = two_txn_wal();
    let idx = scan(&wal);
    // A reader that recorded mxFrame = 2 (first commit) must see txn A only.
    let snap = 2;
    assert_eq!(idx.resolve(1, snap), Some(1)); // page1 = the txn-A version
    assert_eq!(idx.resolve(2, snap), Some(2));
    assert_eq!(idx.resolve(3, snap), None); // page3 does not exist yet
    assert_eq!(idx.db_size_at(snap), 2);
    assert_eq!(idx.resolve_page_data(&wal, 1, snap).unwrap(), Some(&page(0x11)[..]));
}

#[test]
fn uncommitted_trailing_frame_is_not_visible() {
    // Append a valid but uncommitted frame after the last commit (an interrupted
    // transaction): it must not be part of any snapshot.
    let mut b = WalBuilder::new(WalHeader::new(PS, 0xaa, 0xbb, 0, false));
    b.append_frame(1, 0, &page(0x11)).unwrap();
    b.append_frame(2, 2, &page(0x22)).unwrap(); // commit at frame 2
    b.append_frame(2, 0, &page(0x99)).unwrap(); // frame 3: new page2, NO commit after
    let wal = b.into_bytes();

    let idx = scan(&wal);
    assert_eq!(idx.n_valid_frames(), 3, "frame 3 is a valid frame");
    assert_eq!(idx.mx_frame(), 2, "but mxFrame stays at the last commit");
    // Reads never see frame 3's page2; they see the committed frame 2 version.
    assert_eq!(idx.resolve(2, idx.mx_frame()), Some(2));
    assert_eq!(idx.resolve_page_data(&wal, 2, idx.mx_frame()).unwrap(), Some(&page(0x22)[..]));
    // The next writer continues the checksum chain from the last commit (frame 2),
    // overwriting the leftover frame 3.
    assert_eq!(idx.running_checksum_for_append(), idx.frame(2).unwrap().checksum);
}

#[test]
fn corrupting_a_middle_frame_truncates_the_valid_prefix() {
    let wal = two_txn_wal();
    // Flip one byte in frame 3's page data. Frame 3 and everything after it become
    // invalid; the scan stops before frame 3.
    let mut corrupt = wal.clone();
    let data_off = minisqlite_wal::frame_page_data_offset(3, PS);
    corrupt[data_off] ^= 0x01;

    let idx = scan(&corrupt);
    assert_eq!(idx.n_valid_frames(), 2, "scan stops at the first invalid frame");
    assert_eq!(idx.mx_frame(), 2);
    // Page 3 (only in the dropped txn B) is now unreachable.
    assert_eq!(idx.resolve(3, idx.mx_frame()), None);
    // Page 1 resolves to its txn-A frame only.
    assert_eq!(idx.resolve(1, idx.mx_frame()), Some(1));
}

#[test]
fn corrupting_the_header_makes_the_wal_empty() {
    let mut wal = two_txn_wal();
    wal[16] ^= 0xff; // flip a salt-1 byte in the header, breaking its checksum
    let idx = scan(&wal);
    assert!(!idx.has_valid_header());
    assert_eq!(idx.mx_frame(), 0);
    assert_eq!(idx.resolve(1, 10), None);
}

#[test]
fn leftover_frames_with_stale_salts_are_ignored() {
    // A reset WAL: header salts (101, 999), two fresh committed frames, then an old
    // leftover frame carrying the pre-reset salts (100, 200). The scan must stop at
    // the stale frame.
    let mut b = WalBuilder::new(WalHeader::new(PS, 101, 999, 1, false));
    b.append_frame(1, 0, &page(0x11)).unwrap();
    b.append_frame(2, 2, &page(0x22)).unwrap();
    let mut wal = b.into_bytes();

    // Craft a leftover frame with the wrong salts (its checksum is irrelevant — the
    // salt mismatch is checked first).
    let stale = FrameHeader { page_no: 5, db_size: 5, salt1: 100, salt2: 200, checksum: (0, 0) };
    wal.extend_from_slice(&stale.serialize());
    wal.extend_from_slice(&page(0x55));

    let idx = scan(&wal);
    assert_eq!(idx.n_valid_frames(), 2, "stale-salt frame is not counted");
    assert_eq!(idx.mx_frame(), 2);
    assert_eq!(idx.resolve(5, idx.mx_frame()), None);
}

#[test]
fn empty_wal_variants() {
    // Valid header, zero frames.
    let header = WalHeader::new(PS, 7, 8, 0, false);
    let wal = WalBuilder::new(header).into_bytes();
    let idx = scan(&wal);
    assert!(idx.has_valid_header());
    assert!(idx.is_empty());
    assert_eq!(idx.mx_frame(), 0);
    assert_eq!(idx.resolve(1, 0), None);
    assert_eq!(idx.running_checksum_for_append(), header.checksum);

    // No header at all (garbage / too short).
    let idx = scan(&[0u8; 8]);
    assert!(!idx.has_valid_header());
    assert_eq!(idx.mx_frame(), 0);
    let idx = scan(&[]);
    assert!(!idx.has_valid_header());
}

#[test]
fn reset_invalidates_pre_reset_frames() {
    use minisqlite_wal::reset_header;

    // Original WAL with old salts.
    let orig = WalHeader::new(PS, 100, 200, 0, false);
    let mut b = WalBuilder::new(orig);
    b.append_frame(1, 1, &page(0x11)).unwrap();
    let old_frames = &b.as_bytes()[minisqlite_wal::WAL_HEADER_SIZE..].to_vec();

    // Reset: new header with salt1+1, a new salt2, seq+1.
    let reset = reset_header(&orig, 0x1234_5678);
    assert_eq!(reset.salt1, 101);
    assert_ne!(reset.salt2, orig.salt2);
    assert_eq!(reset.checkpoint_seq, 1);

    // A WAL that reused the file: new (reset) header, but the old frame bytes still
    // sit after it (reset need not truncate). They must fail validation now because
    // their salts no longer match the header.
    let mut reused = reset.serialize().to_vec();
    reused.extend_from_slice(old_frames);
    let idx = scan(&reused);
    assert_eq!(idx.n_valid_frames(), 0, "pre-reset frames are invalidated by salts");
    assert_eq!(idx.mx_frame(), 0);

    // Fresh frames written under the reset header validate normally.
    let mut b2 = WalBuilder::new(reset);
    b2.append_frame(1, 1, &page(0xcc)).unwrap();
    let idx2 = scan(b2.as_bytes());
    assert_eq!(idx2.mx_frame(), 1);
    assert_eq!(idx2.resolve_page_data(b2.as_bytes(), 1, 1).unwrap(), Some(&page(0xcc)[..]));
}

#[test]
fn big_endian_wal_round_trips_end_to_end() {
    // The same two-transaction scenario but with big-endian checksums (magic
    // 0x377f0683). Exercises the BE checksum path through the writer (WalBuilder),
    // the scanner, and the reader, plus corruption detection.
    let mut b = WalBuilder::new(WalHeader::new(PS, 0xaa, 0xbb, 0, true));
    b.append_frame(1, 0, &page(0x11)).unwrap();
    b.append_frame(2, 2, &page(0x22)).unwrap();
    b.append_frame(1, 3, &page(0x33)).unwrap();
    let wal = b.into_bytes();
    assert_eq!(&wal[0..4], &minisqlite_wal::WAL_MAGIC_BE.to_be_bytes());

    let idx = scan(&wal);
    assert!(idx.big_endian());
    assert_eq!(idx.mx_frame(), 3);
    assert_eq!(idx.resolve(1, 3), Some(3));
    assert_eq!(idx.resolve_page_data(&wal, 1, 3).unwrap(), Some(&page(0x33)[..]));

    // A single flipped byte still invalidates under BE checksums.
    let mut corrupt = wal.clone();
    corrupt[minisqlite_wal::frame_page_data_offset(3, PS)] ^= 0x01;
    assert_eq!(scan(&corrupt).mx_frame(), 2);
}

#[test]
fn unrecognized_file_format_version_is_rejected() {
    // A header whose checksum is valid but whose file-format version is not
    // WAL_FILE_FORMAT must be treated as an unusable WAL (empty), matching SQLite
    // recovery which rejects a wrong version even when the header checksum passes.
    let mut h = WalHeader::new(PS, 1, 2, 0, false);
    h.file_format = 9_999_999;
    h.checksum = h.compute_checksum(); // self-consistent header, only the version is "wrong"
    assert!(h.verify_checksum());

    let idx = scan(&h.serialize());
    assert!(!idx.has_valid_header(), "wrong version ⇒ unusable header");
    assert_eq!(idx.mx_frame(), 0);
}

/// A deterministic LCG so the property sweep is reproducible without an RNG crate.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes constants.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn in_range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next_u32() % (hi - lo + 1)
    }
}

/// Reference `FindFrame(P, M)` by linear scan over the committed prefix: the largest
/// frame ≤ min(M, mxFrame) carrying page P. For a commit-boundary M this is exactly
/// the reader's answer; for any M it defines the index's contract, so the binary
/// search must match it everywhere.
fn resolve_ref(pages: &[u32], mx_frame: u32, page: u32, snapshot_mx: u32) -> Option<u32> {
    let hi = snapshot_mx.min(mx_frame);
    (1..=hi).rev().find(|&f| pages[(f - 1) as usize] == page)
}

#[test]
fn resolve_matches_reference_over_the_space() {
    for seed in 0..8u64 {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let mut b = WalBuilder::new(WalHeader::new(PS, 1 + seed as u32, 2 + seed as u32, 0, false));
        let n_frames = 40u32;
        let mut pages: Vec<u32> = Vec::new();
        for _ in 0..n_frames {
            let pgno = rng.in_range(1, 8);
            // ~1 in 3 frames is a commit; commit frames carry a db-size.
            let commit = rng.in_range(0, 2) == 0;
            let db_size = if commit { rng.in_range(1, 8) } else { 0 };
            b.append_frame(pgno, db_size, &page((pgno & 0xff) as u8)).unwrap();
            pages.push(pgno);
        }
        let wal = b.into_bytes();
        let idx = scan(&wal);
        assert_eq!(idx.n_valid_frames(), n_frames);
        let mx = idx.mx_frame();

        // Exhaustive sweep over the (page, snapshot) space for this WAL.
        for pgno in 0..=10u32 {
            for snap in 0..=(n_frames + 2) {
                assert_eq!(
                    idx.resolve(pgno, snap),
                    resolve_ref(&pages, mx, pgno, snap),
                    "seed={seed} page={pgno} snap={snap}"
                );
            }
        }
    }
}
