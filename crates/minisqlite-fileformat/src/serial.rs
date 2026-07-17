//! Record (row) format and serial-type codec (fileformat2 §2.1). A record is a
//! header of "serial type" varints followed by the column values. The serial type
//! encodes both the datatype and the on-disk byte length of a column; small
//! constants (NULL, 0, 1) live entirely in the serial type with a zero-length
//! body.
//!
//! Serial type codes:
//!   0 NULL · 1 i8 · 2 i16 · 3 i24 · 4 i32 · 5 i48 · 6 i64 · 7 f64 ·
//!   8 int 0 · 9 int 1 · 10,11 reserved · N>=12 even => BLOB (N-12)/2 bytes ·
//!   N>=13 odd => TEXT (N-13)/2 bytes.
//! All integers are big-endian two's-complement; the float is big-endian IEEE-754.

use crate::text_encoding::TextEncoding;
use crate::varint::{read_varint, varint_len, write_varint};
use minisqlite_types::Value;

// Two's-complement bounds for the sized integer serial types. A value picks the
// smallest serial type whose range contains it, matching how SQLite records the
// minimal encoding (0 and 1 collapse further into serial types 8/9).
const I8_MIN: i64 = -128;
const I8_MAX: i64 = 127;
const I16_MIN: i64 = -32_768;
const I16_MAX: i64 = 32_767;
const I24_MIN: i64 = -8_388_608;
const I24_MAX: i64 = 8_388_607;
const I32_MIN: i64 = -2_147_483_648;
const I32_MAX: i64 = 2_147_483_647;
const I48_MIN: i64 = -140_737_488_355_328;
const I48_MAX: i64 = 140_737_488_355_327;

/// The serial type SQLite writes for `v` inside a record, with TEXT measured in
/// UTF-8 bytes. A convenience wrapper over [`serial_type_of_enc`] for the many
/// UTF-8-by-construction callers (tests, index numeric keys); a writer targeting a
/// UTF-16 database (§1.3.13) must use [`serial_type_of_enc`] so the recorded byte
/// length matches the UTF-16 body it will write.
pub fn serial_type_of(v: &Value) -> u64 {
    serial_type_of_enc(v, TextEncoding::Utf8)
}

/// The serial type SQLite writes for `v` inside a record when the database stores
/// TEXT in encoding `enc` (§1.3.13). Integers use the smallest sized type that holds
/// them, and the constants 0 and 1 use the bodyless serial types 8 and 9 (schema
/// format 4, the default for new files). Only TEXT depends on `enc` — the recorded
/// byte length is the length of the value once transcoded to `enc` (UTF-16 stores two
/// bytes per code unit, so a code point outside the BMP is four bytes, §2.1); every
/// other class ignores `enc` (a BLOB is raw bytes regardless of the text encoding).
pub fn serial_type_of_enc(v: &Value, enc: TextEncoding) -> u64 {
    match v {
        Value::Null => 0,
        Value::Integer(i) => {
            let i = *i;
            if i == 0 {
                8
            } else if i == 1 {
                9
            } else if (I8_MIN..=I8_MAX).contains(&i) {
                1
            } else if (I16_MIN..=I16_MAX).contains(&i) {
                2
            } else if (I24_MIN..=I24_MAX).contains(&i) {
                3
            } else if (I32_MIN..=I32_MAX).contains(&i) {
                4
            } else if (I48_MIN..=I48_MAX).contains(&i) {
                5
            } else {
                6
            }
        }
        Value::Real(_) => 7,
        Value::Text(s) => 13 + 2 * text_encoded_len(s, enc) as u64,
        Value::Blob(b) => 12 + 2 * b.len() as u64,
    }
}

/// The number of bytes `s` occupies when stored in text encoding `enc` (§1.3.13):
/// its UTF-8 byte length as-is, or `2 *` its UTF-16 code-unit count (two bytes per
/// unit; a supplementary-plane code point is a surrogate pair, so four bytes). This
/// is the single source for both the serial-type length and the body write, so the
/// record header and body can never disagree on how long a UTF-16 TEXT column is.
fn text_encoded_len(s: &str, enc: TextEncoding) -> usize {
    match enc {
        TextEncoding::Utf8 => s.len(),
        TextEncoding::Utf16le | TextEncoding::Utf16be => 2 * s.encode_utf16().count(),
    }
}

/// Append `s`'s bytes in text encoding `enc` (§1.3.13) to `out`: the UTF-8 bytes
/// verbatim, or each UTF-16 code unit laid down in the encoding's byte order. The
/// byte count always equals [`text_encoded_len`]`(s, enc)`, so it matches the serial
/// type recorded for the column.
fn write_text_encoded(s: &str, enc: TextEncoding, out: &mut Vec<u8>) {
    match enc {
        TextEncoding::Utf8 => out.extend_from_slice(s.as_bytes()),
        TextEncoding::Utf16le => {
            for u in s.encode_utf16() {
                out.extend_from_slice(&u.to_le_bytes());
            }
        }
        TextEncoding::Utf16be => {
            for u in s.encode_utf16() {
                out.extend_from_slice(&u.to_be_bytes());
            }
        }
    }
}

/// Number of body bytes a serial type occupies. Reserved codes 10 and 11 never
/// appear in a well-formed database file; they are reported as zero-length.
pub fn serial_type_payload_len(serial: u64) -> usize {
    match serial {
        0 | 8 | 9 | 10 | 11 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 6,
        6 | 7 => 8,
        // Even N>=12: (N-12)/2 ; odd N>=13: (N-13)/2. Integer division of (N-12)
        // by 2 yields both (floor drops the +1 for odd N).
        n => ((n - 12) / 2) as usize,
    }
}

/// Decode the value of serial type `serial` from `buf` interpreting TEXT as UTF-8.
/// Thin wrapper over [`read_serial_value_enc`] for callers that are UTF-8 by
/// construction (the engine writes UTF-8; index-key numeric decode and the record
/// tests never touch a UTF-16 body). A caller reading a database whose header
/// declares UTF-16 must use [`read_serial_value_enc`] with that encoding.
pub fn read_serial_value(serial: u64, buf: &[u8]) -> Value {
    read_serial_value_enc(serial, buf, TextEncoding::Utf8)
}

/// Decode the value of serial type `serial` from `buf`, which must hold at least
/// `serial_type_payload_len(serial)` bytes. A TEXT serial type (odd `N >= 13`) is
/// interpreted in `enc` — the database's text encoding (fileformat2 §1.3.13, §2.1) —
/// and transcoded to the engine's internal UTF-8 `String`; every other serial type
/// ignores `enc`. Invalid sequences are replaced rather than panicking.
///
/// PRECONDITION: `buf.len() >= serial_type_payload_len(serial)`. The in-crate
/// caller (`decode_record_into_enc`) always slices exactly that many bytes; a direct
/// caller passing a shorter slice indexes out of bounds. The `debug_assert` makes
/// that misuse loud in tests/debug while the release signature stays infallible
/// (`-> Value`, as the record contract requires).
pub fn read_serial_value_enc(serial: u64, buf: &[u8], enc: TextEncoding) -> Value {
    debug_assert!(
        buf.len() >= serial_type_payload_len(serial),
        "read_serial_value_enc: buffer of {} bytes is shorter than serial {serial}'s {} payload bytes",
        buf.len(),
        serial_type_payload_len(serial)
    );
    match serial {
        0 => Value::Null,
        1 => Value::Integer(buf[0] as i8 as i64),
        2 => Value::Integer(i16::from_be_bytes([buf[0], buf[1]]) as i64),
        3 => {
            let raw = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
            // Sign-extend the 24-bit two's-complement value.
            let signed = ((raw << 8) as i32) >> 8;
            Value::Integer(signed as i64)
        }
        4 => Value::Integer(i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as i64),
        5 => {
            let mut raw: u64 = 0;
            for &b in &buf[..6] {
                raw = (raw << 8) | b as u64;
            }
            // Sign-extend the 48-bit two's-complement value.
            let signed = ((raw << 16) as i64) >> 16;
            Value::Integer(signed)
        }
        6 => Value::Integer(i64::from_be_bytes(buf[..8].try_into().unwrap())),
        7 => Value::Real(f64::from_be_bytes(buf[..8].try_into().unwrap())),
        8 => Value::Integer(0),
        9 => Value::Integer(1),
        10 | 11 => Value::Null,
        n => {
            let len = ((n - 12) / 2) as usize;
            let bytes = &buf[..len];
            if n & 1 == 0 {
                Value::Blob(bytes.to_vec())
            } else {
                // TEXT: the stored bytes are in the database's declared encoding
                // (§1.3.13); transcode to the engine's internal UTF-8 `String`.
                Value::Text(enc.decode_text(bytes))
            }
        }
    }
}

/// Append the body bytes for `v` to `out`, TEXT in UTF-8. Thin wrapper over
/// [`write_serial_value_enc`]. The serial type (written into the record header) is
/// derived from the same [`serial_type_of`] classification, so the header and body
/// always agree on length.
pub fn write_serial_value(v: &Value, out: &mut Vec<u8>) {
    write_serial_value_enc(v, TextEncoding::Utf8, out);
}

/// Append the body bytes for `v` to `out`, TEXT in encoding `enc` (§1.3.13). The
/// serial type recorded for the column (via [`serial_type_of_enc`]) uses the same
/// `enc`, so the header and body agree on length.
pub fn write_serial_value_enc(v: &Value, enc: TextEncoding, out: &mut Vec<u8>) {
    write_value_body_enc(v, serial_type_of_enc(v, enc), enc, out);
}

/// Append `v`'s body bytes given its already-computed `serial`, TEXT in encoding
/// `enc` (§1.3.13). Only integers depend on the serial (it picks the width) and only
/// TEXT depends on `enc`; the other classes ignore both. Kept separate so
/// [`encode_record_into_enc`] reuses the serial it classified for the header rather
/// than re-deriving it per value.
fn write_value_body_enc(v: &Value, serial: u64, enc: TextEncoding, out: &mut Vec<u8>) {
    match v {
        Value::Null => {}
        Value::Integer(i) => write_int_body(*i, serial, out),
        Value::Real(f) => out.extend_from_slice(&f.to_be_bytes()),
        Value::Text(s) => write_text_encoded(s, enc, out),
        Value::Blob(b) => out.extend_from_slice(b),
    }
}

/// Write the big-endian body for an integer whose serial type is already chosen.
/// Serial types 8/9 (the constants 0/1) have no body.
fn write_int_body(i: i64, serial: u64, out: &mut Vec<u8>) {
    match serial {
        8 | 9 => {}
        1 => out.push(i as u8),
        2 => out.extend_from_slice(&(i as i16).to_be_bytes()),
        3 => out.extend_from_slice(&(i as i32).to_be_bytes()[1..4]),
        4 => out.extend_from_slice(&(i as i32).to_be_bytes()),
        5 => out.extend_from_slice(&i.to_be_bytes()[2..8]),
        6 => out.extend_from_slice(&i.to_be_bytes()),
        _ => unreachable!("serial_type_of never yields {serial} for an integer"),
    }
}

/// Decode a full record into `out` interpreting TEXT as UTF-8. Thin wrapper over
/// [`decode_record_into_enc`] for the many UTF-8-by-construction callers (the write
/// path, and every read of a database this engine authored). A reader of a
/// UTF-16 database (per its header, §1.3.13) must use [`decode_record_into_enc`].
pub fn decode_record_into(buf: &[u8], out: &mut Vec<Value>) {
    decode_record_into_enc(buf, TextEncoding::Utf8, out);
}

/// Decode a full record into `out` (cleared first), interpreting TEXT columns in the
/// database's text encoding `enc` (§1.3.13). Yields exactly the values physically
/// present: a record with fewer serial types than the table has columns leaves the
/// trailing columns absent, and the caller substitutes the column default (NULL
/// unless a schema default applies). A truncated body stops decoding rather than
/// panicking, so a corrupt cell yields a short row instead of a crash.
pub fn decode_record_into_enc(buf: &[u8], enc: TextEncoding, out: &mut Vec<Value>) {
    out.clear();
    let Some((header_len, first_type_off)) = read_varint(buf) else {
        return;
    };
    let header_len = header_len as usize;
    let header_end = header_len.min(buf.len());
    let mut type_off = first_type_off;
    // The body begins immediately after the full header (whose declared length
    // includes the size varint itself).
    let mut body_off = header_len;
    while type_off < header_end {
        let Some((serial, n)) = read_varint(&buf[type_off..header_end]) else {
            break;
        };
        type_off += n;
        let len = serial_type_payload_len(serial);
        if len == 0 {
            out.push(read_serial_value_enc(serial, &[], enc));
            continue;
        }
        // Slice `[body_off .. body_off+len]` without forming `body_off+len`, which
        // would overflow usize (a debug panic) when a corrupt record declares a
        // header length near u64::MAX. Take the tail from body_off, then len bytes —
        // failing closed on both a too-large offset and a too-large length. This
        // keeps decode_record byte-for-byte equivalent to RecordCursor.
        let Some(body) = buf.get(body_off..).and_then(|rest| rest.get(..len)) else {
            break; // truncated body: stop at the last complete value.
        };
        out.push(read_serial_value_enc(serial, body, enc));
        body_off += len;
    }
}

/// Decode a full record, allocating the result vector (TEXT as UTF-8).
pub fn decode_record(buf: &[u8]) -> Vec<Value> {
    decode_record_enc(buf, TextEncoding::Utf8)
}

/// Decode a full record in text encoding `enc` (§1.3.13), allocating the result
/// vector. The encoding-aware companion of [`decode_record`].
pub fn decode_record_enc(buf: &[u8], enc: TextEncoding) -> Vec<Value> {
    let mut out = Vec::new();
    decode_record_into_enc(buf, enc, &mut out);
    out
}

/// Encode `values` into a record — a self-describing header of serial-type varints
/// (prefixed by the total header length) followed by the packed value bodies —
/// APPENDING to `out` (it is not cleared). A hot write path reuses one buffer
/// across rows by clearing it between them. Each value's serial type is classified
/// exactly once and reused for both the header and the body, and the bytes are
/// written straight into `out` with no intermediate header buffer.
pub fn encode_record_into(values: &[Value], out: &mut Vec<u8>) {
    encode_record_into_enc(values, TextEncoding::Utf8, out);
}

/// Encode `values` into a record with TEXT stored in encoding `enc` (§1.3.13),
/// APPENDING to `out`. The encoding-aware companion of [`encode_record_into`]; a
/// writer targeting a UTF-16 database uses this so every TEXT column — data rows,
/// index keys, and the `sqlite_schema` rows themselves — is laid down in the byte
/// order real sqlite expects for that database. With `enc == Utf8` the output is
/// byte-identical to [`encode_record_into`].
pub fn encode_record_into_enc(values: &[Value], enc: TextEncoding, out: &mut Vec<u8>) {
    let mut serials: Vec<u64> = Vec::with_capacity(values.len());
    let mut header_body_len = 0usize;
    let mut body_len = 0usize;
    for v in values {
        let serial = serial_type_of_enc(v, enc);
        header_body_len += varint_len(serial);
        body_len += serial_type_payload_len(serial);
        serials.push(serial);
    }
    // The leading length varint counts itself, so its own width feeds back into
    // the total. Solve the fixed point (converges in at most a couple of steps
    // because varint_len grows far slower than its argument).
    let mut len_width = 1usize;
    for _ in 0..9 {
        let candidate = varint_len((header_body_len + len_width) as u64);
        if candidate == len_width {
            break;
        }
        len_width = candidate;
    }
    let header_len = header_body_len + len_width;

    out.reserve(header_len + body_len);
    write_varint(header_len as u64, out);
    for &serial in &serials {
        write_varint(serial, out);
    }
    for (v, &serial) in values.iter().zip(&serials) {
        write_value_body_enc(v, serial, enc, out);
    }
}

/// Encode `values` into a fresh record byte vector (TEXT in UTF-8). See
/// [`encode_record_into`] for the buffer-reusing form for hot write paths.
pub fn encode_record(values: &[Value]) -> Vec<u8> {
    encode_record_enc(values, TextEncoding::Utf8)
}

/// Encode `values` into a fresh record byte vector with TEXT stored in encoding
/// `enc` (§1.3.13). The encoding-aware companion of [`encode_record`].
pub fn encode_record_enc(values: &[Value], enc: TextEncoding) -> Vec<u8> {
    let mut out = Vec::new();
    encode_record_into_enc(values, enc, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v_eq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Integer(x), Value::Integer(y)) => x == y,
            (Value::Real(x), Value::Real(y)) => x.to_bits() == y.to_bits(),
            (Value::Text(x), Value::Text(y)) => x == y,
            (Value::Blob(x), Value::Blob(y)) => x == y,
            _ => false,
        }
    }

    fn assert_rows_eq(a: &[Value], b: &[Value]) {
        assert_eq!(a.len(), b.len(), "row length {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!(v_eq(x, y), "value {x:?} != {y:?}");
        }
    }

    #[test]
    fn serial_type_selection() {
        assert_eq!(serial_type_of(&Value::Null), 0);
        assert_eq!(serial_type_of(&Value::Integer(0)), 8);
        assert_eq!(serial_type_of(&Value::Integer(1)), 9);
        assert_eq!(serial_type_of(&Value::Integer(2)), 1);
        assert_eq!(serial_type_of(&Value::Integer(-1)), 1);
        assert_eq!(serial_type_of(&Value::Integer(127)), 1);
        assert_eq!(serial_type_of(&Value::Integer(128)), 2);
        assert_eq!(serial_type_of(&Value::Integer(-128)), 1);
        assert_eq!(serial_type_of(&Value::Integer(-129)), 2);
        assert_eq!(serial_type_of(&Value::Integer(32_767)), 2);
        assert_eq!(serial_type_of(&Value::Integer(32_768)), 3);
        assert_eq!(serial_type_of(&Value::Integer(I24_MAX)), 3);
        assert_eq!(serial_type_of(&Value::Integer(I24_MAX + 1)), 4);
        assert_eq!(serial_type_of(&Value::Integer(I32_MAX)), 4);
        assert_eq!(serial_type_of(&Value::Integer(I32_MAX + 1)), 5);
        assert_eq!(serial_type_of(&Value::Integer(I48_MAX)), 5);
        assert_eq!(serial_type_of(&Value::Integer(I48_MAX + 1)), 6);
        assert_eq!(serial_type_of(&Value::Integer(i64::MAX)), 6);
        assert_eq!(serial_type_of(&Value::Integer(i64::MIN)), 6);
        assert_eq!(serial_type_of(&Value::Real(1.5)), 7);
        assert_eq!(serial_type_of(&Value::Text("abc".into())), 13 + 2 * 3);
        assert_eq!(serial_type_of(&Value::Text(String::new())), 13);
        assert_eq!(serial_type_of(&Value::Blob(vec![0; 4])), 12 + 2 * 4);
        assert_eq!(serial_type_of(&Value::Blob(Vec::new())), 12);
    }

    #[test]
    fn payload_len_matches_serial() {
        assert_eq!(serial_type_payload_len(0), 0);
        assert_eq!(serial_type_payload_len(1), 1);
        assert_eq!(serial_type_payload_len(2), 2);
        assert_eq!(serial_type_payload_len(3), 3);
        assert_eq!(serial_type_payload_len(4), 4);
        assert_eq!(serial_type_payload_len(5), 6);
        assert_eq!(serial_type_payload_len(6), 8);
        assert_eq!(serial_type_payload_len(7), 8);
        assert_eq!(serial_type_payload_len(8), 0);
        assert_eq!(serial_type_payload_len(9), 0);
        assert_eq!(serial_type_payload_len(12), 0); // empty blob
        assert_eq!(serial_type_payload_len(13), 0); // empty text
        assert_eq!(serial_type_payload_len(14), 1); // 1-byte blob
        assert_eq!(serial_type_payload_len(15), 1); // 1-byte text
        assert_eq!(serial_type_payload_len(100), (100 - 12) / 2);
        assert_eq!(serial_type_payload_len(101), (101 - 13) / 2);
    }

    #[test]
    fn each_value_body_length_and_roundtrip() {
        let samples = [
            Value::Null,
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-1),
            Value::Integer(127),
            Value::Integer(-128),
            Value::Integer(200),
            Value::Integer(-200),
            Value::Integer(32_767),
            Value::Integer(-32_768),
            Value::Integer(8_388_607),
            Value::Integer(-8_388_608),
            Value::Integer(2_147_483_647),
            Value::Integer(-2_147_483_648),
            Value::Integer(I48_MAX),
            Value::Integer(I48_MIN),
            Value::Integer(i64::MAX),
            Value::Integer(i64::MIN),
            Value::Real(0.0),
            Value::Real(-0.0),
            Value::Real(3.141592653589793),
            Value::Real(f64::INFINITY),
            Value::Text("hello".into()),
            Value::Text(String::new()),
            Value::Text("λ→utf8".into()),
            Value::Blob(vec![0, 1, 2, 253, 254, 255]),
            Value::Blob(Vec::new()),
        ];
        for v in &samples {
            let serial = serial_type_of(v);
            let mut body = Vec::new();
            write_serial_value(v, &mut body);
            assert_eq!(
                body.len(),
                serial_type_payload_len(serial),
                "body length disagrees for {v:?} (serial {serial})"
            );
            let got = read_serial_value(serial, &body);
            assert!(v_eq(&got, v), "serial round-trip {v:?} -> {got:?}");
        }
    }

    #[test]
    fn record_roundtrip_mixed() {
        let row = vec![
            Value::Null,
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-42),
            Value::Integer(1_000_000_000_000),
            Value::Real(2.5),
            Value::Text("mixed".into()),
            Value::Blob(vec![9, 8, 7]),
        ];
        let encoded = encode_record(&row);
        let decoded = decode_record(&encoded);
        assert_rows_eq(&decoded, &row);
    }

    #[test]
    fn record_header_layout_is_documented() {
        // One 8-bit integer 5 and the text "hi" (2 bytes -> serial 13+2*2 = 17).
        // Header = [len=3][serial 1][serial 17]; body = [0x05, 'h', 'i']. The
        // header length (3) counts the size varint, the two serial-type varints.
        let row = vec![Value::Integer(5), Value::Text("hi".into())];
        let encoded = encode_record(&row);
        // header length varint (value 3), serial 1, serial 17, then body 0x05 'h' 'i'
        assert_eq!(encoded[0], 3, "header length includes the size varint");
        assert_eq!(encoded[1], 1, "first serial type: i8");
        assert_eq!(encoded[2], 17, "second serial type: text of 2 bytes");
        assert_eq!(&encoded[3..], &[0x05, b'h', b'i']);
        assert_rows_eq(&decode_record(&encoded), &row);
    }

    #[test]
    fn empty_record_and_all_zero_length() {
        // All bodyless: NULL, 0, 1, empty text, empty blob.
        let row = vec![
            Value::Null,
            Value::Integer(0),
            Value::Integer(1),
            Value::Text(String::new()),
            Value::Blob(Vec::new()),
        ];
        let encoded = encode_record(&row);
        // Body is empty; every value lives in the header.
        let header_len = encoded[0] as usize;
        assert_eq!(header_len, encoded.len(), "no body bytes for zero-length row");
        assert_rows_eq(&decode_record(&encoded), &row);

        // A zero-column record is just a 1-byte header of value 1.
        let empty = encode_record(&[]);
        assert_eq!(empty, vec![1]);
        assert!(decode_record(&empty).is_empty());
    }

    #[test]
    fn short_record_yields_only_present_values() {
        // A 3-column table stored a 2-value record (e.g. after ADD COLUMN). Decode
        // must surface exactly the two present values; the caller pads the third.
        let row = vec![Value::Integer(10), Value::Text("x".into())];
        let encoded = encode_record(&row);
        let decoded = decode_record(&encoded);
        assert_eq!(decoded.len(), 2);
        assert_rows_eq(&decoded, &row);
    }

    #[test]
    fn truncated_body_stops_at_last_complete_value() {
        // Three i16 columns (2 body bytes each). If the body is cut after the first
        // value, decode must stop at the last complete value — not panic or read
        // past the buffer (a corrupt/truncated cell must fail soft).
        let row = vec![
            Value::Integer(0x1111),
            Value::Integer(0x2222),
            Value::Integer(0x3333),
        ];
        let full = encode_record(&row);
        let (header_len, _) = read_varint(&full).unwrap();
        // Keep the whole header plus exactly one value's body (2 bytes).
        let truncated = &full[..header_len as usize + 2];
        let decoded = decode_record(truncated);
        assert_eq!(decoded.len(), 1, "stops at the one complete value");
        assert!(matches!(decoded[0], Value::Integer(0x1111)));
    }

    #[test]
    fn corrupt_huge_header_len_does_not_panic() {
        // Leading length varint = u64::MAX ([0xFF; 9]), then a serial-type byte
        // (0x04 = i32). The declared body offset is usize::MAX, so a naive
        // `body_off + len` slice overflows and debug-panics. decode must uphold its
        // "stop, don't panic" contract: fail closed to a short (empty) row.
        let corrupt = [0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x04];
        assert!(decode_record(&corrupt).is_empty());
    }

    #[test]
    fn decoders_agree_on_corrupt_inputs() {
        // decode_record and RecordCursor must yield identical value sequences for
        // ANY byte buffer, not just well-formed records. record_cursor's
        // equivalence_prng_sweep only feeds valid encode_record output, so it cannot
        // catch a one-sided change to a FAIL-CLOSED branch (truncated header/body,
        // corrupt or huge declared length) — exactly where this round's body_off+len
        // overflow lived, in BOTH duplicated walks. Sweeping adversarial buffers
        // (including 9-byte-0xFF, i.e. header_len = u64::MAX, prefixes) pins the two
        // walks in lockstep on those branches, so a future reintroduction on only
        // one side is caught here.
        use crate::record_cursor::RecordCursor;
        let mut state: u64 = 0x0bad_f00d_dead_beef;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let n = (next() % 20) as usize;
            let mut buf: Vec<u8> = (0..n).map(|_| (next() & 0xff) as u8).collect();
            // A quarter of the time, force the corrupt huge-header-length branch.
            if next() % 4 == 0 {
                let mut c = vec![0xFFu8; 9];
                c.extend_from_slice(&buf);
                buf = c;
            }
            let via_decode = decode_record(&buf);
            let via_cursor: Vec<Value> = RecordCursor::new(&buf)
                .map(|(s, b)| read_serial_value(s, b))
                .collect();
            assert_eq!(via_decode.len(), via_cursor.len(), "len mismatch on {buf:?}");
            for (a, b) in via_decode.iter().zip(&via_cursor) {
                assert!(v_eq(a, b), "value mismatch on {buf:?}: {a:?} != {b:?}");
            }
        }
    }

    #[test]
    fn reserved_serial_types_10_11() {
        // Codes 10 and 11 never appear in a well-formed file; they are treated as
        // zero-length and decode to NULL rather than panicking.
        assert_eq!(serial_type_payload_len(10), 0);
        assert_eq!(serial_type_payload_len(11), 0);
        assert!(matches!(read_serial_value(10, &[]), Value::Null));
        assert!(matches!(read_serial_value(11, &[]), Value::Null));
    }

    #[test]
    fn encode_record_into_appends_and_matches_encode_record() {
        // The buffer-reusing form appends and produces identical bytes to the
        // allocating form; reuse across rows must not corrupt earlier content.
        let row_a = vec![Value::Integer(0x2222), Value::Text("a".into())];
        let row_b = vec![Value::Null, Value::Blob(vec![9, 9])];
        assert_eq!(encode_record(&row_a), {
            let mut buf = Vec::new();
            encode_record_into(&row_a, &mut buf);
            buf
        });
        // Appends (does not clear): a preset prefix is preserved.
        let mut buf = vec![0xEE];
        encode_record_into(&row_a, &mut buf);
        assert_eq!(buf[0], 0xEE);
        assert_eq!(&buf[1..], encode_record(&row_a).as_slice());
        // Reuse across rows by clearing between them.
        buf.clear();
        encode_record_into(&row_b, &mut buf);
        assert_eq!(buf, encode_record(&row_b));
    }

    #[test]
    fn large_header_needs_two_byte_length_varint() {
        // Enough columns that the serial-type header exceeds 127 bytes, forcing the
        // self-referential header-length varint to two bytes.
        let row: Vec<Value> = (0..200).map(|_| Value::Integer(7)).collect();
        let encoded = encode_record(&row);
        let (header_len, off) = read_varint(&encoded).unwrap();
        assert!(off >= 2, "header length varint should be multi-byte");
        assert_eq!(header_len as usize, off + row.len(), "200 one-byte serials + width");
        assert_rows_eq(&decode_record(&encoded), &row);
    }

    #[test]
    fn value_bodies_are_byte_exact_big_endian() {
        // Lock the exact on-disk bytes (not just a self round-trip): every sized
        // integer is big-endian two's-complement, the float is big-endian IEEE-754.
        fn body(v: &Value) -> Vec<u8> {
            let mut out = Vec::new();
            write_serial_value(v, &mut out);
            out
        }
        assert_eq!(body(&Value::Integer(2)), vec![0x02]); // serial 1
        assert_eq!(body(&Value::Integer(-1)), vec![0xFF]);
        assert_eq!(body(&Value::Integer(-2)), vec![0xFE]);
        assert_eq!(body(&Value::Integer(0x1234)), vec![0x12, 0x34]); // serial 2
        assert_eq!(body(&Value::Integer(-256)), vec![0xFF, 0x00]);
        assert_eq!(body(&Value::Integer(0x12_3456)), vec![0x12, 0x34, 0x56]); // serial 3
        assert_eq!(body(&Value::Integer(-0x80_0000)), vec![0x80, 0x00, 0x00]); // i24 min
        assert_eq!(body(&Value::Integer(0x1234_5678)), vec![0x12, 0x34, 0x56, 0x78]); // serial 4
        assert_eq!(
            body(&Value::Integer(0x1234_5678_9ABC)),
            vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC] // serial 5 (48-bit)
        );
        assert_eq!(
            body(&Value::Integer(0x0102_0304_0506_0708)),
            vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] // serial 6
        );
        // 1.0f64 = 0x3FF0000000000000 big-endian.
        assert_eq!(
            body(&Value::Real(1.0)),
            vec![0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        // Text/blob bodies are the raw bytes with no length prefix or NUL.
        assert_eq!(body(&Value::Text("Ab".into())), vec![b'A', b'b']);
        assert_eq!(body(&Value::Blob(vec![0xDE, 0xAD])), vec![0xDE, 0xAD]);
        // The zero/one constants and NULL carry no body bytes.
        assert!(body(&Value::Integer(0)).is_empty());
        assert!(body(&Value::Integer(1)).is_empty());
        assert!(body(&Value::Null).is_empty());
    }

    #[test]
    fn integer_body_prng_roundtrip() {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let i = state as i64;
            let v = Value::Integer(i);
            let serial = serial_type_of(&v);
            let mut body = Vec::new();
            write_serial_value(&v, &mut body);
            assert_eq!(body.len(), serial_type_payload_len(serial));
            assert!(v_eq(&read_serial_value(serial, &body), &v), "int {i}");
        }
    }
}
