//! Zero-copy iteration over a record's columns (fileformat2 §2.1).
//!
//! [`crate::serial::decode_record`] eagerly materializes every column into an
//! owned [`Value`], which copies TEXT/BLOB bytes out of the page. On a hot scan
//! the executor often needs only one column (a `WHERE` predicate, a projected
//! subset), so a `RecordCursor` instead walks the record header lazily and yields
//! each column's serial type paired with a BORROWED slice of its body bytes —
//! allocating nothing and copying nothing until the caller decodes a value it
//! actually wants (via [`crate::serial::read_serial_value`]).
//!
//! The cursor is defined to be exactly equivalent to `decode_record`: mapping
//! [`read_serial_value`](crate::serial::read_serial_value) over its items yields
//! the same sequence `decode_record` produces, for every input (well-formed or
//! truncated). Like `decode_record` it is tolerant of a short record — it stops
//! at the last complete column rather than erroring — so the caller applies the
//! same "missing trailing columns take their default" rule (§2.1).

use crate::serial::serial_type_payload_len;
use crate::varint::read_varint;

/// A lazy, borrowing reader over the columns of one record. Yields `(serial_type,
/// body_slice)` pairs in column order; the slice borrows from the record buffer.
#[derive(Debug, Clone)]
pub struct RecordCursor<'a> {
    buf: &'a [u8],
    /// Declared header length (the offset at which column bodies begin).
    header_len: usize,
    /// End of the readable header region: `min(header_len, buf.len())`. Serial-type
    /// varints are read only within `[.., header_end)`.
    header_end: usize,
    /// Offset of the next serial-type varint within the header.
    type_off: usize,
    /// Offset of the next column body within the buffer.
    body_off: usize,
}

impl<'a> RecordCursor<'a> {
    /// Start a cursor over the record encoded at the front of `buf`. Never fails:
    /// a truncated header (or empty buffer) yields a cursor that produces no
    /// columns, matching `decode_record`'s tolerance.
    pub fn new(buf: &'a [u8]) -> RecordCursor<'a> {
        match read_varint(buf) {
            Some((header_len, first_type_off)) => {
                let header_len = header_len as usize;
                RecordCursor {
                    buf,
                    header_len,
                    // The declared body starts at header_len; the header's serial
                    // types are read only up to what the buffer actually holds.
                    header_end: header_len.min(buf.len()),
                    type_off: first_type_off,
                    body_off: header_len,
                }
            }
            // No decodable header length: an empty cursor (type_off >= header_end).
            None => RecordCursor { buf, header_len: 0, header_end: 0, type_off: 0, body_off: 0 },
        }
    }

    /// The declared record header length in bytes (including the size varint
    /// itself), i.e. the offset at which the column bodies begin.
    pub fn header_len(&self) -> usize {
        self.header_len
    }
}

impl<'a> Iterator for RecordCursor<'a> {
    /// Each item is a column's `(serial_type, borrowed_body_bytes)`. Zero-length
    /// serial types (NULL, 0/1 constants, empty TEXT/BLOB) yield an empty slice.
    type Item = (u64, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.type_off >= self.header_end {
            return None;
        }
        let (serial, n) = match read_varint(&self.buf[self.type_off..self.header_end]) {
            Some(pair) => pair,
            // A serial-type varint that does not terminate within the header is a
            // truncated header: stop cleanly (mirrors decode_record).
            None => {
                self.type_off = self.header_end;
                return None;
            }
        };
        self.type_off += n;
        let len = serial_type_payload_len(serial);
        if len == 0 {
            return Some((serial, &[]));
        }
        // Slice `[body_off .. body_off+len]` without ever forming `body_off+len`:
        // `body_off` is the *declared* (unclamped) header length, so on a corrupt
        // record it can be near usize::MAX and the addition would overflow (a debug
        // panic — violating this cursor's "stop, don't panic" contract). Taking the
        // tail first then `len` bytes fails closed on both a too-large `body_off`
        // and a too-large `len`.
        match self.buf.get(self.body_off..).and_then(|rest| rest.get(..len)) {
            Some(body) => {
                self.body_off += len;
                Some((serial, body))
            }
            // Body runs past the record: stop at the last complete column.
            None => {
                self.type_off = self.header_end;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serial::{decode_record, encode_record, read_serial_value};
    use minisqlite_types::Value;

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

    #[test]
    fn cursor_matches_decode_record() {
        let row = vec![
            Value::Null,
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-42),
            Value::Integer(1_000_000_000_000),
            Value::Real(2.5),
            Value::Text("mixed".into()),
            Value::Blob(vec![9, 8, 7]),
            Value::Text(String::new()),
            Value::Blob(Vec::new()),
        ];
        let buf = encode_record(&row);
        let via_cursor: Vec<Value> = RecordCursor::new(&buf)
            .map(|(serial, body)| read_serial_value(serial, body))
            .collect();
        let via_decode = decode_record(&buf);
        assert_eq!(via_cursor.len(), via_decode.len());
        for (a, b) in via_cursor.iter().zip(&via_decode) {
            assert!(v_eq(a, b), "{a:?} != {b:?}");
        }
        // And both equal the source row.
        for (a, b) in via_cursor.iter().zip(&row) {
            assert!(v_eq(a, b));
        }
    }

    #[test]
    fn header_len_reports_body_start() {
        // [len=3][serial 1][serial 17] then body — header length is 3.
        let row = vec![Value::Integer(5), Value::Text("hi".into())];
        let buf = encode_record(&row);
        let cur = RecordCursor::new(&buf);
        assert_eq!(cur.header_len(), 3);
    }

    #[test]
    fn bodies_borrow_from_the_input() {
        let row = vec![Value::Text("borrowed".into()), Value::Blob(vec![1, 2, 3, 4])];
        let buf = encode_record(&row);
        let buf_range = buf.as_ptr_range();
        for (_serial, body) in RecordCursor::new(&buf) {
            if body.is_empty() {
                continue;
            }
            let body_range = body.as_ptr_range();
            assert!(
                body_range.start >= buf_range.start && body_range.end <= buf_range.end,
                "column body must be a sub-slice of the record buffer (zero-copy)"
            );
        }
    }

    #[test]
    fn can_skip_to_a_single_column_without_decoding_others() {
        // Read only column index 2 (a Text), never materializing the rest.
        let row = vec![
            Value::Integer(7),
            Value::Blob(vec![0xAA; 32]),
            Value::Text("target".into()),
            Value::Real(1.0),
        ];
        let buf = encode_record(&row);
        let mut cur = RecordCursor::new(&buf);
        let (serial, body) = cur.nth(2).expect("third column present");
        assert!(v_eq(&read_serial_value(serial, body), &Value::Text("target".into())));
    }

    #[test]
    fn empty_and_truncated_records() {
        // Zero-column record: a lone header-length varint of value 1.
        let empty = encode_record(&[]);
        assert_eq!(RecordCursor::new(&empty).count(), 0);

        // Empty buffer: no columns, no panic.
        assert_eq!(RecordCursor::new(&[]).count(), 0);

        // Truncated body: declares an i32 (serial 4, 4 bytes) but only 2 body bytes
        // are present. The cursor stops at the last complete column (here: none),
        // exactly as decode_record does.
        let truncated = [0x02u8, 0x04, 0x11, 0x22]; // header len 2, serial 4, 2 body bytes
        let via_cursor: Vec<Value> = RecordCursor::new(&truncated)
            .map(|(s, b)| read_serial_value(s, b))
            .collect();
        assert_eq!(via_cursor.len(), decode_record(&truncated).len());
        assert!(via_cursor.is_empty());
    }

    #[test]
    fn corrupt_huge_header_len_does_not_panic() {
        // A record whose leading length varint decodes to u64::MAX ([0xFF; 9]),
        // followed by a serial-type byte (0x04 = i32, 4-byte body). The declared
        // body_off is then usize::MAX, so a naive `body_off + len` slice overflows
        // and debug-panics. The cursor must instead fail closed: no columns, no
        // panic (its stated contract, and equivalence with decode_record which
        // handles the same input the same way).
        let corrupt = [0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x04];
        assert_eq!(RecordCursor::new(&corrupt).count(), 0);
        assert_eq!(RecordCursor::new(&corrupt).count(), decode_record(&corrupt).len());
    }

    #[test]
    fn equivalence_prng_sweep() {
        // Property: for randomly generated rows, the cursor and decode_record
        // produce identical value sequences. This pins the cursor to the reference
        // codec over the whole input space, not a hand-picked case.
        let mut state: u64 = 0xda7a_1057_c0ffee11;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..5_000 {
            let ncols = (next() % 8) as usize;
            let row: Vec<Value> = (0..ncols)
                .map(|_| match next() % 6 {
                    0 => Value::Null,
                    1 => Value::Integer(next() as i64),
                    2 => Value::Real(f64::from_bits(next())),
                    3 => {
                        let len = (next() % 12) as usize;
                        Value::Text("a".repeat(len))
                    }
                    4 => {
                        let len = (next() % 12) as usize;
                        Value::Blob(vec![(next() & 0xff) as u8; len])
                    }
                    _ => Value::Integer((next() % 3) as i64), // exercises 0/1 constants
                })
                .collect();
            let buf = encode_record(&row);
            let via_cursor: Vec<Value> = RecordCursor::new(&buf)
                .map(|(s, b)| read_serial_value(s, b))
                .collect();
            let via_decode = decode_record(&buf);
            assert_eq!(via_cursor.len(), via_decode.len());
            for (a, b) in via_cursor.iter().zip(&via_decode) {
                assert!(v_eq(a, b), "row {row:?}: {a:?} != {b:?}");
            }
        }
    }
}
