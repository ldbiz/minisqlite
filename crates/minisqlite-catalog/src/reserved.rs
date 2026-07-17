//! The reserved schema-name predicate: whether a schema-object name uses the
//! SQLite-internal `sqlite_` prefix (fileformat2 §2.6, lang_createtable.html §2).
//!
//! SQLite reserves every name beginning `sqlite_` (case-insensitive) for its own
//! internal objects — `sqlite_schema`/`sqlite_master`, `sqlite_sequence`,
//! `sqlite_stat1`, the `sqlite_autoindex_*` auto-indexes. A user `CREATE` may not use
//! the prefix, and internal callers (ANALYZE's stat scan, VACUUM's copy) use this to
//! tell an internal object from a user one.
//!
//! This is the shared home for the check, on purpose: the predicate was previously
//! spelled independently in several crates, and one copy drifted to the `name[..7]`
//! STRING-slice form which PANICS ("byte index 7 is not a char boundary") on a
//! `>= 7`-byte name whose byte 7 falls inside a multibyte UTF-8 character (a Unicode
//! table name). This form slices over `as_bytes()`, which is panic-free because
//! `sqlite_` is pure ASCII and a 7-byte ASCII prefix compare never splits a UTF-8
//! scalar, so callers need not respell it. (Note: `schemacatalog.rs` still checks the
//! prefix inline via `key.starts_with("sqlite_")` on already-`norm`-folded keys — those
//! are panic-free and equivalent; nothing yet forces every site through this one fn, so
//! routing them here is a possible future consolidation, not a guarantee this enforces.)

/// Whether `name` is a reserved SQLite-internal schema-object name: it begins with the
/// `sqlite_` prefix, compared case-insensitively (ASCII). Panic-free for every input,
/// including names whose byte 7 falls inside a multibyte UTF-8 character.
pub fn is_reserved_schema_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() >= 7 && bytes[..7].eq_ignore_ascii_case(b"sqlite_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reserved_prefix_case_insensitively() {
        assert!(is_reserved_schema_name("sqlite_sequence"));
        assert!(is_reserved_schema_name("sqlite_stat1"));
        assert!(is_reserved_schema_name("sqlite_autoindex_t_1"));
        assert!(is_reserved_schema_name("SQLITE_master"));
        assert!(is_reserved_schema_name("SqLiTe_whatever"));
        assert!(is_reserved_schema_name("sqlite_"), "exactly the 7-byte prefix is reserved");
    }

    #[test]
    fn rejects_user_names() {
        assert!(!is_reserved_schema_name(""));
        assert!(!is_reserved_schema_name("sqlit"), "too short");
        assert!(!is_reserved_schema_name("sqlite"), "6 bytes, missing the underscore");
        assert!(!is_reserved_schema_name("t"));
        assert!(!is_reserved_schema_name("users"));
        assert!(!is_reserved_schema_name("my_sqlite_log"), "prefix not at the start");
    }

    #[test]
    fn does_not_panic_on_a_multibyte_char_at_the_prefix_boundary() {
        // A `>= 7`-byte name whose byte 7 is inside a multibyte UTF-8 char would PANIC
        // under `name[..7]` STRING slicing. The byte-slice form must classify it (a
        // user name) without panicking — this is the regression guard.
        // "aaaa" (4) + U+1F600 😀 (4 bytes) = 8 bytes; byte 7 is mid-emoji.
        let emoji = "aaaa\u{1F600}";
        assert_eq!(emoji.len(), 8);
        assert!(!emoji.is_char_boundary(7), "byte 7 must be mid-char for this test to bite");
        assert!(!is_reserved_schema_name(emoji));
        // 6 ASCII + 'ñ' (2 bytes) = 8 bytes; the char spans bytes 6..8, so byte 7 is mid-char.
        let ntilde = "aaaaaañ";
        assert_eq!(ntilde.len(), 8);
        assert!(!ntilde.is_char_boundary(7));
        assert!(!is_reserved_schema_name(ntilde));
        // A reserved name may still carry multibyte content AFTER the ASCII prefix.
        assert!(is_reserved_schema_name("sqlite_ñ"));
    }
}
