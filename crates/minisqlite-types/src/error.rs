//! Engine errors and SQLite result-code vocabulary.
//!
//! What matters is success vs. error and, for a few cases, the error *kind* —
//! not the exact wording. The four `Error` variants are stable (the facade
//! re-exports `Error`); this module keeps them and adds a typed `ConstraintKind`,
//! the SQLite primary/extended result-code constants, and constructor helpers so
//! callers raise errors of a consistent shape.

/// Engine errors. What is observable is success vs. error (and, for a few cases,
/// the error *kind*) — not exact wording.
///
/// The variant set is stable (single-field tuples): the facade re-exports `Error`
/// and downstream code may pattern-match these, so their names and shapes stay
/// stable. Kind detail beyond these four is carried by [`ConstraintKind`] and the
/// result-code constants below, and folded into the message at construction time.
#[derive(Debug)]
pub enum Error {
    /// Malformed, ambiguous, or unsupported SQL.
    Sql(String),
    /// The on-disk file is not valid SQLite, or could not be read/written in the
    /// official format.
    Format(String),
    /// Underlying I/O failure.
    Io(String),
    /// A constraint (PRIMARY KEY / UNIQUE / NOT NULL / CHECK / FK) was violated.
    Constraint(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Sql(m) => write!(f, "sql error: {m}"),
            Error::Format(m) => write!(f, "format error: {m}"),
            Error::Io(m) => write!(f, "io error: {m}"),
            Error::Constraint(m) => write!(f, "constraint error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// I/O failures map to `Error::Io` so the `?` operator works across the storage
/// layer without a manual `.map_err` at every call site.
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// SQLite primary result codes (the low byte of an extended code). Only the ones
/// this engine can produce are named; the numeric values match SQLite so a caller
/// mapping to the C-API surface reports the same code the reference does.
///
/// See `spec/sqlite-doc/rescode.html`.
pub mod code {
    pub const OK: i32 = 0;
    pub const ERROR: i32 = 1;
    pub const PERM: i32 = 3;
    pub const BUSY: i32 = 5;
    pub const NOMEM: i32 = 7;
    pub const READONLY: i32 = 8;
    pub const IOERR: i32 = 10;
    pub const CORRUPT: i32 = 11;
    pub const FULL: i32 = 13;
    pub const CANTOPEN: i32 = 14;
    pub const CONSTRAINT: i32 = 19;
    pub const MISMATCH: i32 = 20;
    pub const MISUSE: i32 = 21;
    pub const RANGE: i32 = 25;
    pub const NOTADB: i32 = 26;

    /// Build an extended code from a primary code and a high-byte subcode, exactly
    /// as SQLite does: `primary | (sub << 8)`.
    pub const fn extended(primary: i32, sub: i32) -> i32 {
        primary | (sub << 8)
    }

    // Extended CONSTRAINT codes (the subcodes SQLite assigns each constraint
    // class). Kept in sync with `ConstraintKind::extended_code`.
    pub const CONSTRAINT_CHECK: i32 = extended(CONSTRAINT, 1);
    pub const CONSTRAINT_FOREIGNKEY: i32 = extended(CONSTRAINT, 3);
    pub const CONSTRAINT_NOTNULL: i32 = extended(CONSTRAINT, 5);
    pub const CONSTRAINT_PRIMARYKEY: i32 = extended(CONSTRAINT, 6);
    pub const CONSTRAINT_UNIQUE: i32 = extended(CONSTRAINT, 8);
    pub const CONSTRAINT_ROWID: i32 = extended(CONSTRAINT, 10);
    pub const CONSTRAINT_TRIGGER: i32 = extended(CONSTRAINT, 7);
}

/// The class of constraint that failed. Used to shape a consistent error message
/// and to expose the SQLite extended result code for callers that report it. The
/// kind is chosen at the point the violation is detected (the executor), where the
/// context to name it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    PrimaryKey,
    Unique,
    NotNull,
    Check,
    ForeignKey,
    RowId,
    Trigger,
}

impl ConstraintKind {
    /// The SQLite extended result code for this constraint class (all share the
    /// `SQLITE_CONSTRAINT` primary code).
    pub fn extended_code(self) -> i32 {
        match self {
            ConstraintKind::Check => code::CONSTRAINT_CHECK,
            ConstraintKind::ForeignKey => code::CONSTRAINT_FOREIGNKEY,
            ConstraintKind::NotNull => code::CONSTRAINT_NOTNULL,
            ConstraintKind::PrimaryKey => code::CONSTRAINT_PRIMARYKEY,
            ConstraintKind::Unique => code::CONSTRAINT_UNIQUE,
            ConstraintKind::RowId => code::CONSTRAINT_ROWID,
            ConstraintKind::Trigger => code::CONSTRAINT_TRIGGER,
        }
    }

    /// The phrase SQLite puts at the start of the failure message, e.g.
    /// `"UNIQUE constraint failed"`. Wording is not asserted, but a stable,
    /// reference-shaped message keeps logs legible.
    pub fn message_prefix(self) -> &'static str {
        match self {
            ConstraintKind::PrimaryKey => "PRIMARY KEY constraint failed",
            ConstraintKind::Unique => "UNIQUE constraint failed",
            ConstraintKind::NotNull => "NOT NULL constraint failed",
            ConstraintKind::Check => "CHECK constraint failed",
            ConstraintKind::ForeignKey => "FOREIGN KEY constraint failed",
            ConstraintKind::RowId => "ROWID constraint failed",
            ConstraintKind::Trigger => "trigger constraint failed",
        }
    }
}

impl Error {
    /// A malformed / unsupported SQL error (`SQLITE_ERROR`).
    pub fn sql(msg: impl Into<String>) -> Error {
        Error::Sql(msg.into())
    }

    /// An on-disk format error (`SQLITE_CORRUPT` / `SQLITE_NOTADB` family).
    pub fn format(msg: impl Into<String>) -> Error {
        Error::Format(msg.into())
    }

    /// An I/O error (`SQLITE_IOERR`).
    pub fn io(msg: impl Into<String>) -> Error {
        Error::Io(msg.into())
    }

    /// A typed constraint violation. `detail` names the offending object (e.g.
    /// `"t.c"`); it is appended to the kind's phrase to form the message. The
    /// extended code is available via `kind.extended_code()`.
    pub fn constraint(kind: ConstraintKind, detail: impl AsRef<str>) -> Error {
        let detail = detail.as_ref();
        let msg = if detail.is_empty() {
            kind.message_prefix().to_string()
        } else {
            format!("{}: {}", kind.message_prefix(), detail)
        };
        Error::Constraint(msg)
    }

    /// The SQLite primary result code for this error, for callers that report a
    /// numeric code. Constraint sub-classification lives in [`ConstraintKind`].
    pub fn primary_code(&self) -> i32 {
        match self {
            Error::Sql(_) => code::ERROR,
            Error::Format(_) => code::CORRUPT,
            Error::Io(_) => code::IOERR,
            Error::Constraint(_) => code::CONSTRAINT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_codes_match_sqlite_layout() {
        // primary | (sub << 8): documented SQLite values.
        assert_eq!(code::CONSTRAINT_UNIQUE, 19 | (8 << 8));
        assert_eq!(code::CONSTRAINT_PRIMARYKEY, 19 | (6 << 8));
        assert_eq!(code::CONSTRAINT_NOTNULL, 19 | (5 << 8));
        assert_eq!(code::CONSTRAINT_CHECK, 19 | (1 << 8));
        assert_eq!(code::CONSTRAINT_FOREIGNKEY, 19 | (3 << 8));
        // Low byte of every extended constraint code is the primary code.
        assert_eq!(code::CONSTRAINT_UNIQUE & 0xff, code::CONSTRAINT);
    }

    #[test]
    fn constraint_constructor_carries_code_and_message() {
        let e = Error::constraint(ConstraintKind::Unique, "t.c");
        assert_eq!(e.primary_code(), code::CONSTRAINT);
        assert_eq!(ConstraintKind::Unique.extended_code(), code::CONSTRAINT_UNIQUE);
        assert!(matches!(&e, Error::Constraint(m) if m == "UNIQUE constraint failed: t.c"));
    }

    #[test]
    fn empty_detail_omits_separator() {
        let e = Error::constraint(ConstraintKind::NotNull, "");
        assert!(matches!(&e, Error::Constraint(m) if m == "NOT NULL constraint failed"));
    }

    #[test]
    fn primary_codes_by_variant() {
        assert_eq!(Error::sql("x").primary_code(), code::ERROR);
        assert_eq!(Error::format("x").primary_code(), code::CORRUPT);
        assert_eq!(Error::io("x").primary_code(), code::IOERR);
    }

    #[test]
    fn io_error_conversion() {
        let e: Error = std::io::Error::new(std::io::ErrorKind::Other, "disk").into();
        assert!(matches!(e, Error::Io(_)));
    }
}
