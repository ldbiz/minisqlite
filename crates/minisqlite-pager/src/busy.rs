//! BUSY signalling for the WAL write lock.
//!
//! When a second connection tries to write while another holds the single WAL write
//! lock, real SQLite returns `SQLITE_BUSY` (`minisqlite_types::code::BUSY` == 5) with
//! the message `"database is locked"`. `minisqlite_types::Error` has four *pinned*
//! variants (`Sql`/`Format`/`Io`/`Constraint`) re-exported by the facade and no
//! dedicated `Busy` variant — that vocabulary is owned by `minisqlite-types` — so a
//! write-lock conflict here rides on [`Error::Io`] carrying exactly that canonical
//! message. [`is_busy`] recognizes such an error and is the behavioral seam a caller
//! uses to tell a *retryable* BUSY from a genuine I/O failure, rather than
//! string-sniffing scattered across call sites. That seam is currently exercised only
//! by this crate's own tests; the engine and facade do not call it.
//!
//! This delivers BUSY *behaviorally* — the write errors, it is retryable, and it
//! carries sqlite's `"database is locked"` message. It does NOT yet deliver the
//! matching *numeric* result code: [`Error::primary_code`] maps every `Error::Io` to
//! `IOERR` (10), so a WAL write-lock conflict surfaces with primary code 10, not
//! `code::BUSY` (5). Mapping this message to `code::BUSY` is a pending change in
//! `minisqlite-types` (teaching `Error::primary_code` to recognize it); it is
//! separately owned there and not applied in the current tree.

use minisqlite_types::Error;

/// The exact message a WAL write-lock conflict carries. Equal to `sqlite3`'s
/// `SQLITE_BUSY` text so it matches real sqlite, and stable so
/// [`is_busy`] can recognize it.
pub const BUSY_MESSAGE: &str = "database is locked";

/// Construct the BUSY error returned when the WAL write lock is already held.
pub(crate) fn busy_error() -> Error {
    Error::Io(BUSY_MESSAGE.to_string())
}

/// True if `err` is a WAL write-lock BUSY. This is the behavioral seam a caller uses
/// to tell a *retryable* lock conflict (back off and retry the transaction) from an
/// error that is terminal for the statement. It matches the canonical
/// `"database is locked"` message carried on [`Error::Io`]; the numeric primary
/// result code is still `IOERR` (10) — see the module docs for the pending
/// `code::BUSY` (5) mapping.
pub fn is_busy(err: &Error) -> bool {
    matches!(err, Error::Io(msg) if msg == BUSY_MESSAGE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_types::code;

    #[test]
    fn busy_error_is_recognized() {
        let e = busy_error();
        assert!(is_busy(&e));
    }

    #[test]
    fn other_io_errors_are_not_busy() {
        assert!(!is_busy(&Error::Io("disk full".into())));
        assert!(!is_busy(&Error::Format("bad".into())));
        assert!(!is_busy(&Error::Sql("nope".into())));
    }

    #[test]
    fn busy_code_constant_is_five() {
        // Pin the documented mapping so a drift in the shared vocabulary is caught.
        assert_eq!(code::BUSY, 5);
    }

    #[test]
    fn busy_message_is_the_exact_wire_string() {
        // Pin sqlite's exact SQLITE_BUSY text. Real sqlite emits this exact
        // string, and `busy_error_is_recognized` cannot catch a drift
        // here because it round-trips through BUSY_MESSAGE on both sides.
        assert_eq!(BUSY_MESSAGE, "database is locked");
    }

    #[test]
    fn busy_error_primary_code_is_currently_ioerr() {
        // Doc-rot tripwire, pinning the CURRENT (divergent-from-sqlite) numeric code on
        // purpose: a WAL BUSY rides on `Error::Io`, which `primary_code` maps to IOERR
        // (10), not sqlite's BUSY (5). When the pending `minisqlite-types` change makes a
        // WAL BUSY report `code::BUSY` (5), THIS assertion AND this module's
        // pending-mapping docs must be updated together — the failing test is the
        // lockstep reminder to keep them in sync, not a defect to silence.
        assert_eq!(busy_error().primary_code(), code::IOERR);
    }
}
