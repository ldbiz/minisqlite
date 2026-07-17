//! The database text encoding (fileformat2 §1.3.13, header offset 56) and the
//! conversion of a stored TEXT body into the engine's internal UTF-8 `String`.
//!
//! Every TEXT value in a database file is stored in ONE encoding chosen when the
//! database is created: UTF-8, UTF-16 little-endian, or UTF-16 big-endian (§1.3.13).
//! The record codec (`serial.rs`) records only the byte LENGTH of a TEXT column
//! (serial `N` odd `>= 13` ⇒ `(N-13)/2` bytes, §2.1); it is THIS type that says how
//! those bytes decode into characters. The engine is UTF-8 internally, so a UTF-16
//! database's text is transcoded to UTF-8 on read, and — via the encoding-aware
//! `serial.rs` writers (`encode_record_enc` and friends) — transcoded back to the
//! database's encoding on write, so a UTF-16 database the engine creates is
//! byte-for-byte readable by real sqlite.

use crate::header::{DatabaseHeader, TEXT_ENCODING_UTF16BE, TEXT_ENCODING_UTF16LE};

/// The text encoding a database file's TEXT values are stored in (§1.3.13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextEncoding {
    /// UTF-8 (header code 1, and the legacy/unset code 0). The engine's internal form,
    /// so decoding is a validated pass-through.
    #[default]
    Utf8,
    /// UTF-16 little-endian (header code 2).
    Utf16le,
    /// UTF-16 big-endian (header code 3).
    Utf16be,
}

impl TextEncoding {
    /// Map the header's offset-56 code (§1.3.13) to a typed encoding: 2 ⇒ UTF-16le,
    /// 3 ⇒ UTF-16be, everything else ⇒ UTF-8. Code 1 is UTF-8 and code 0 is the unset
    /// legacy default which SQLite also treats as UTF-8, so the catch-all is faithful
    /// (SQLite only ever writes 1/2/3) rather than a silent swallow of a real variant.
    pub fn from_code(code: u32) -> TextEncoding {
        match code {
            TEXT_ENCODING_UTF16LE => TextEncoding::Utf16le,
            TEXT_ENCODING_UTF16BE => TextEncoding::Utf16be,
            _ => TextEncoding::Utf8,
        }
    }

    /// Decode a TEXT column body — exactly the `(N-13)/2` stored bytes (§2.1) — into an
    /// owned UTF-8 `String`.
    ///
    /// * UTF-8: validated lossily, invalid byte sequences becoming U+FFFD (unchanged
    ///   from the engine's prior TEXT behavior — a corrupt cell never panics).
    /// * UTF-16: the bytes are read as 2-byte code units in the encoding's byte order,
    ///   and `char::decode_utf16` combines surrogate PAIRS into their code point while
    ///   mapping a lone/unpaired surrogate to U+FFFD. A well-formed UTF-16 body is
    ///   always even-length (§2.1: the length is `2 * n` code units); a trailing odd
    ///   byte is corruption and contributes one U+FFFD rather than being dropped
    ///   silently, so the loss is visible in the value rather than off the books.
    pub fn decode_text(self, bytes: &[u8]) -> String {
        match self {
            TextEncoding::Utf8 => String::from_utf8_lossy(bytes).into_owned(),
            TextEncoding::Utf16le => decode_utf16(bytes, u16::from_le_bytes),
            TextEncoding::Utf16be => decode_utf16(bytes, u16::from_be_bytes),
        }
    }
}

impl DatabaseHeader {
    /// This database's text encoding as a typed [`TextEncoding`] (offset 56, §1.3.13).
    /// The one place the raw `text_encoding` code is interpreted, so the read path can
    /// thread a single value instead of re-deciding the mapping at each decode site.
    pub fn text_encoding_kind(&self) -> TextEncoding {
        TextEncoding::from_code(self.text_encoding)
    }
}

/// Decode UTF-16 bytes into a `String`, reading each 2-byte code unit with `to_u16`
/// (little- or big-endian). Surrogate pairs are combined and lone surrogates replaced
/// (`char::decode_utf16`); a trailing odd byte (corrupt — a UTF-16 body is even-length)
/// yields a final U+FFFD so the corruption surfaces in the value.
fn decode_utf16(bytes: &[u8], to_u16: fn([u8; 2]) -> u16) -> String {
    let chunks = bytes.chunks_exact(2);
    // A leftover byte means the body was not a whole number of UTF-16 code units.
    let has_odd_tail = !chunks.remainder().is_empty();
    let mut s: String = char::decode_utf16(chunks.map(|c| to_u16([c[0], c[1]])))
        .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect();
    if has_odd_tail {
        s.push(char::REPLACEMENT_CHARACTER);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_mapping_covers_all_documented_values() {
        assert_eq!(TextEncoding::from_code(1), TextEncoding::Utf8);
        assert_eq!(TextEncoding::from_code(2), TextEncoding::Utf16le);
        assert_eq!(TextEncoding::from_code(3), TextEncoding::Utf16be);
        // 0 is the unset/legacy default (UTF-8); any other value is not a real code and
        // also degrades to UTF-8 rather than panicking.
        assert_eq!(TextEncoding::from_code(0), TextEncoding::Utf8);
        assert_eq!(TextEncoding::from_code(99), TextEncoding::Utf8);
        assert_eq!(TextEncoding::default(), TextEncoding::Utf8);
    }

    #[test]
    fn utf8_passthrough_matches_prior_behavior() {
        assert_eq!(TextEncoding::Utf8.decode_text(b"hello"), "hello");
        assert_eq!(TextEncoding::Utf8.decode_text(&[]), "");
        // Invalid UTF-8 is replaced, not a panic (a corrupt cell must fail soft).
        assert_eq!(TextEncoding::Utf8.decode_text(&[0xFF, 0xFE]), "\u{FFFD}\u{FFFD}");
    }

    /// Encode a `&str` to UTF-16 bytes in the given order — an INDEPENDENT reference
    /// for the decode tests (never the crate's own encoder, which does not exist for
    /// UTF-16 write). Uses `str::encode_utf16` (the std code-unit iterator) then lays
    /// each unit down in the requested byte order.
    fn to_utf16(s: &str, big_endian: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for u in s.encode_utf16() {
            let b = if big_endian { u.to_be_bytes() } else { u.to_le_bytes() };
            out.extend_from_slice(&b);
        }
        out
    }

    #[test]
    fn utf16_roundtrips_bmp_and_supplementary() {
        // BMP text, a non-Latin script, and a supplementary-plane emoji (which is a
        // SURROGATE PAIR in UTF-16 — the case the naive "one u16 = one char" decode
        // gets wrong). Empty string included.
        for s in ["", "hi", "café", "Пример", "日本語", "a\u{1F600}z", "\u{10FFFF}"] {
            let le = to_utf16(s, false);
            let be = to_utf16(s, true);
            assert_eq!(TextEncoding::Utf16le.decode_text(&le), s, "LE {s:?}");
            assert_eq!(TextEncoding::Utf16be.decode_text(&be), s, "BE {s:?}");
            // A UTF-16 body is even-length (§2.1).
            assert_eq!(le.len() % 2, 0, "LE body even for {s:?}");
            assert_eq!(be.len() % 2, 0, "BE body even for {s:?}");
        }
    }

    #[test]
    fn utf16_byte_order_is_honored() {
        // 'A' = U+0041. LE bytes are [0x41, 0x00]; BE bytes are [0x00, 0x41]. Decoding
        // the SAME bytes under the wrong order would not yield "A", so this pins that
        // the byte order actually drives the decode (not a symmetric accident).
        assert_eq!(TextEncoding::Utf16le.decode_text(&[0x41, 0x00]), "A");
        assert_eq!(TextEncoding::Utf16be.decode_text(&[0x00, 0x41]), "A");
        // Cross-reading is wrong (a control char U+4100 vs "A") — proves order matters.
        assert_ne!(TextEncoding::Utf16be.decode_text(&[0x41, 0x00]), "A");
    }

    #[test]
    fn utf16_surrogate_pair_decodes_to_one_codepoint() {
        // U+1F600 (😀): UTF-16 surrogate pair D83D DE00. Assert the pair combines into
        // the single code point, not two replacement chars.
        let le = vec![0x3D, 0xD8, 0x00, 0xDE];
        let be = vec![0xD8, 0x3D, 0xDE, 0x00];
        assert_eq!(TextEncoding::Utf16le.decode_text(&le), "\u{1F600}");
        assert_eq!(TextEncoding::Utf16be.decode_text(&be), "\u{1F600}");
    }

    #[test]
    fn utf16_lone_surrogate_becomes_replacement() {
        // A high surrogate with no following low surrogate is invalid; decode must not
        // panic and must surface a replacement char (accounted-for loss).
        let lone_high_be = vec![0xD8, 0x3D]; // D83D with nothing after
        assert_eq!(TextEncoding::Utf16be.decode_text(&lone_high_be), "\u{FFFD}");
    }

    #[test]
    fn utf16_odd_trailing_byte_surfaces_a_replacement() {
        // A corrupt, odd-length UTF-16 body (§2.1 says TEXT is even-length for UTF-16):
        // the whole code units decode, and the leftover byte becomes a visible U+FFFD
        // rather than being silently dropped.
        let bytes = vec![0x41, 0x00, 0x42]; // "A" then a stray 0x42
        assert_eq!(TextEncoding::Utf16le.decode_text(&bytes), "A\u{FFFD}");
    }

    #[test]
    fn header_kind_reads_offset_56() {
        let mut h = DatabaseHeader::default();
        assert_eq!(h.text_encoding_kind(), TextEncoding::Utf8);
        h.text_encoding = TEXT_ENCODING_UTF16LE;
        assert_eq!(h.text_encoding_kind(), TextEncoding::Utf16le);
        h.text_encoding = TEXT_ENCODING_UTF16BE;
        assert_eq!(h.text_encoding_kind(), TextEncoding::Utf16be);
    }
}
