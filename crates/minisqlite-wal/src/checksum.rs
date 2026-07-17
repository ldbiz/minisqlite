//! The WAL checksum (fileformat2 §4.2).
//!
//! The checksum is two 32-bit Fibonacci-weighted running sums over the input
//! interpreted as an even number of 32-bit words. It is *cumulative*: the WAL
//! header seeds from `(0, 0)`, and each frame continues from the previous frame's
//! result (the header result for the first frame). The word byte order is chosen
//! by the WAL header magic — little-endian for `0x377f0682`, big-endian for
//! `0x377f0683` — independently of the fact that every field is *stored* on disk
//! big-endian. The two output words are always stored big-endian regardless.

/// Compute the WAL checksum over `data`, continuing from `seed`.
///
/// `data.len()` MUST be a multiple of 8 (the algorithm consumes 32-bit word
/// pairs); the WAL header slice (24 bytes) and every frame's `first-8-bytes ++
/// page-data` slice (8 + page_size, and page sizes are powers of two ≥ 512)
/// satisfy this by construction. A non-multiple-of-8 length is a caller bug: the
/// trailing partial word is ignored, and a debug build asserts.
///
/// `big_endian` selects the word byte order (true ⇔ header magic `0x377f0683`).
/// The returned `(s0, s1)` is the running value to store (big-endian) or to feed
/// as the next `seed`.
#[inline]
pub fn wal_checksum(seed: (u32, u32), data: &[u8], big_endian: bool) -> (u32, u32) {
    debug_assert!(
        data.len().is_multiple_of(8),
        "wal_checksum input must be a multiple of 8 bytes, got {}",
        data.len()
    );
    let (mut s0, mut s1) = seed;
    let mut chunks = data.chunks_exact(8);
    if big_endian {
        for c in &mut chunks {
            let x0 = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
            let x1 = u32::from_be_bytes([c[4], c[5], c[6], c[7]]);
            s0 = s0.wrapping_add(x0.wrapping_add(s1));
            s1 = s1.wrapping_add(x1.wrapping_add(s0));
        }
    } else {
        for c in &mut chunks {
            let x0 = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let x1 = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            s0 = s0.wrapping_add(x0.wrapping_add(s1));
            s1 = s1.wrapping_add(x1.wrapping_add(s0));
        }
    }
    (s0, s1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-computed vectors. For [1,0,0,0, 2,0,0,0] read little-endian the words
    // are x0=1, x1=2: s0 = 0+(1+0)=1; s1 = 0+(2+1)=3.
    #[test]
    fn le_single_word_pair() {
        assert_eq!(wal_checksum((0, 0), &[1, 0, 0, 0, 2, 0, 0, 0], false), (1, 3));
    }

    // The same numeric words (1, 2) encoded big-endian give the same result.
    #[test]
    fn be_single_word_pair() {
        assert_eq!(wal_checksum((0, 0), &[0, 0, 0, 1, 0, 0, 0, 2], true), (1, 3));
    }

    // Two word pairs [1,2,3,4] little-endian:
    //   i=0: s0=0+(1+0)=1; s1=0+(2+1)=3
    //   i=2: s0=1+(3+3)=7; s1=3+(4+7)=14
    #[test]
    fn le_two_word_pairs() {
        let data = [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0];
        assert_eq!(wal_checksum((0, 0), &data, false), (7, 14));
    }

    // Cumulative seeding: feeding the second pair with the first pair's result must
    // equal computing both pairs in one call.
    #[test]
    fn cumulative_matches_single_call() {
        let a = [1, 0, 0, 0, 2, 0, 0, 0];
        let b = [3, 0, 0, 0, 4, 0, 0, 0];
        let both = [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0];
        let seeded = wal_checksum(wal_checksum((0, 0), &a, false), &b, false);
        assert_eq!(seeded, wal_checksum((0, 0), &both, false));
        assert_eq!(seeded, (7, 14));
    }

    // Byte order genuinely matters for non-symmetric input.
    #[test]
    fn le_and_be_differ_for_multibyte_words() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_ne!(wal_checksum((0, 0), &data, false), wal_checksum((0, 0), &data, true));
    }

    // Empty input is a no-op that returns the seed unchanged.
    #[test]
    fn empty_is_identity() {
        assert_eq!(wal_checksum((5, 9), &[], false), (5, 9));
    }

    // The sums wrap (mod 2^32) rather than overflow-panicking.
    #[test]
    fn wraps_on_overflow() {
        let data = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        let (s0, s1) = wal_checksum((0, 0), &data, false);
        // x0 = x1 = 0xffff_ffff. s0 = 0 + (0xffff_ffff + 0) = 0xffff_ffff.
        // s1 = 0 + (0xffff_ffff + 0xffff_ffff) = 0xffff_fffe (wrapped).
        assert_eq!(s0, 0xffff_ffff);
        assert_eq!(s1, 0xffff_fffe);
    }
}
