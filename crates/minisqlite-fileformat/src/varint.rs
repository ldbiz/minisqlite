//! SQLite "varint" — a big-endian static Huffman encoding of a 64-bit value,
//! 1 to 9 bytes long (fileformat2 §1.6). Every byte except the last of a
//! multi-byte varint has its high bit set as a continuation flag and contributes
//! its low 7 bits, most-significant group first. The exception is the 9th byte:
//! a varint never exceeds 9 bytes, so once eight continuation bytes have been
//! read the ninth byte contributes all 8 of its bits (there is no room for a
//! flag). A value needs the 9-byte form exactly when it does not fit in 8*7=56
//! bits, i.e. when any of bits 56..=63 is set.

/// Mask of bits 56..=63. A value needs the full 9-byte varint iff it intersects
/// this mask (it cannot fit in eight 7-bit groups).
const NINE_BYTE_MASK: u64 = 0xff00_0000_0000_0000;

/// Decode a varint from the front of `buf`. Returns the value and the number of
/// bytes consumed (1..=9), or `None` if `buf` ends before the varint completes.
pub fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    // First up to 8 bytes: 7 payload bits each, high bit = continue.
    for i in 0..8 {
        let byte = *buf.get(i)?;
        result = (result << 7) | (byte & 0x7f) as u64;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
    }
    // Ninth byte: all 8 bits contribute (56 bits so far + 8 = 64).
    let byte = *buf.get(8)?;
    result = (result << 8) | byte as u64;
    Some((result, 9))
}

/// Append the varint encoding of `v` to `out`, returning the number of bytes
/// written (1..=9). The encoding is minimal length except for the 9-byte form,
/// which is used exactly for values that do not fit in 56 bits.
pub fn write_varint(v: u64, out: &mut Vec<u8>) -> usize {
    if v & NINE_BYTE_MASK != 0 {
        // 9-byte form: eight continuation bytes then a full 8-bit final byte.
        let mut tmp = [0u8; 9];
        tmp[8] = v as u8;
        let mut rest = v >> 8;
        for slot in tmp[..8].iter_mut().rev() {
            *slot = (rest as u8 & 0x7f) | 0x80;
            rest >>= 7;
        }
        out.extend_from_slice(&tmp);
        return 9;
    }
    // Emit 7-bit groups least-significant first into a scratch buffer, then
    // reverse. The first group emitted (least significant) is the terminating
    // byte, so its continuation bit is cleared.
    let mut scratch = [0u8; 9];
    let mut n = 0;
    let mut rest = v;
    loop {
        scratch[n] = (rest as u8 & 0x7f) | 0x80;
        n += 1;
        rest >>= 7;
        if rest == 0 {
            break;
        }
    }
    scratch[0] &= 0x7f;
    for &b in scratch[..n].iter().rev() {
        out.push(b);
    }
    n
}

/// Number of bytes `write_varint(v, ..)` will produce, without encoding it.
/// Useful for sizing a record header (whose length varint is self-referential).
pub fn varint_len(v: u64) -> usize {
    if v & NINE_BYTE_MASK != 0 {
        return 9;
    }
    let mut n = 1;
    let mut rest = v >> 7;
    while rest != 0 {
        n += 1;
        rest >>= 7;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: u64, expect_len: usize) {
        let mut buf = Vec::new();
        let written = write_varint(v, &mut buf);
        assert_eq!(written, buf.len(), "write returns bytes written");
        assert_eq!(written, expect_len, "value {v:#x} encodes to {expect_len} bytes");
        assert_eq!(varint_len(v), expect_len, "varint_len matches write for {v:#x}");
        let (got, consumed) = read_varint(&buf).expect("decode");
        assert_eq!(got, v, "round-trip value {v:#x}");
        assert_eq!(consumed, written, "round-trip length {v:#x}");
    }

    #[test]
    fn boundary_lengths() {
        roundtrip(0, 1);
        roundtrip(1, 1);
        roundtrip(127, 1); // 0x7f — largest 1-byte
        roundtrip(128, 2); // smallest 2-byte
        roundtrip(0x3fff, 2); // largest 2-byte (14 bits)
        roundtrip(0x4000, 3);
        roundtrip((1 << 21) - 1, 3);
        roundtrip(1 << 21, 4);
        roundtrip((1 << 28) - 1, 4);
        roundtrip(1 << 28, 5);
        roundtrip((1 << 35) - 1, 5);
        roundtrip(1 << 35, 6);
        roundtrip((1 << 42) - 1, 6);
        roundtrip(1 << 42, 7);
        roundtrip((1 << 49) - 1, 7);
        roundtrip(1 << 49, 8);
        roundtrip((1u64 << 56) - 1, 8); // largest 8-byte
        roundtrip(1u64 << 56, 9); // smallest 9-byte
        roundtrip(u64::MAX, 9);
    }

    #[test]
    fn nine_byte_all_ones() {
        // u64::MAX must be exactly nine 0xff... the eight continuation bytes are
        // 0xff (high bit + seven 1s) and the ninth is 0xff (all bits).
        let mut buf = Vec::new();
        write_varint(u64::MAX, &mut buf);
        assert_eq!(buf, vec![0xff; 9]);
        assert_eq!(read_varint(&buf), Some((u64::MAX, 9)));
    }

    #[test]
    fn known_encodings() {
        // Cross-checked against the spec's group-by-7-bits layout.
        let cases: &[(u64, &[u8])] = &[
            (0x00, &[0x00]),
            (0x7f, &[0x7f]),
            (0x80, &[0x81, 0x00]),
            (0x81, &[0x81, 0x01]),
            (0x100, &[0x82, 0x00]),
            (0x3fff, &[0xff, 0x7f]),
            (0x4000, &[0x81, 0x80, 0x00]),
        ];
        for (v, bytes) in cases {
            let mut buf = Vec::new();
            write_varint(*v, &mut buf);
            assert_eq!(&buf[..], *bytes, "encoding of {v:#x}");
            assert_eq!(read_varint(bytes), Some((*v, bytes.len())));
        }
    }

    #[test]
    fn truncated_returns_none() {
        assert_eq!(read_varint(&[]), None);
        // A stream of continuation bytes with no terminator and < 9 bytes.
        assert_eq!(read_varint(&[0x80, 0x80]), None);
        // Exactly 8 continuation bytes but no 9th byte present.
        assert_eq!(read_varint(&[0x80; 8]), None);
        // 8 continuation bytes + a raw 9th byte is a complete 9-byte varint. The
        // 9th byte contributes all 8 of its bits, so 0x80 here means 128 (not a
        // continuation flag), while the first 8 bytes contribute 0.
        assert_eq!(read_varint(&[0x80; 9]), Some((128, 9)));
        // The genuinely-zero 9-byte encoding ends in a 0x00 ninth byte.
        assert_eq!(
            read_varint(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00]),
            Some((0, 9))
        );
    }

    #[test]
    fn exhaustive_low_and_prng_high() {
        // Exhaust every value up to 3-byte varints (0..2^21), which covers all the
        // 1/2/3-byte length transitions and the shift/mask logic.
        for v in 0u64..(1 << 21) {
            let mut buf = Vec::new();
            let n = write_varint(v, &mut buf);
            assert_eq!(varint_len(v), n);
            assert_eq!(read_varint(&buf), Some((v, n)));
        }
        // Deterministic PRNG sweep over the full 64-bit range for the wide cases.
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut buf = Vec::new();
            let n = write_varint(state, &mut buf);
            assert_eq!(varint_len(state), n);
            let (got, consumed) = read_varint(&buf).unwrap();
            assert_eq!(got, state);
            assert_eq!(consumed, n);
            // Trailing bytes after a complete varint must not be consumed.
            buf.push(0xAB);
            assert_eq!(read_varint(&buf), Some((state, n)));
        }
    }
}
