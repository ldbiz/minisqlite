//! The `PRAGMA wal_checkpoint(...)` vocabulary threaded through the `Pager::checkpoint`
//! seam: the checkpoint MODE the caller selects, and the REPORT the seam returns.
//!
//! These are plain data types (not a competing seam trait) that carry the checkpoint
//! mode down to the WAL store and the `(busy, log, checkpointed)` outcome back up to
//! the engine's `PRAGMA wal_checkpoint` handler, so the mode-specific behavior
//! (pragma.html #pragma_wal_checkpoint) lives once, at the store, rather than being
//! re-derived per caller.

/// The mode a `PRAGMA wal_checkpoint(<arg>)` selects (pragma.html
/// #pragma_wal_checkpoint). Each mode drains committed frames back into the database
/// file; they differ in how hard they try to also RESET the log, and therefore in
/// when they report `busy` (blocked from completing):
///
/// - `Passive` — drain as many frames as possible without waiting for readers or
///   writers; never blocks (`busy` is always 0). The bare `PRAGMA wal_checkpoint`
///   and any unrecognized argument map here.
/// - `Full` — drain EVERY frame; blocked (`busy` = 1) while a writer is active or a
///   reader still pins a snapshot behind the log tail.
/// - `Restart` — `Full` plus reset the log so the next writer starts from its
///   beginning; blocked while ANY reader is still on the log.
/// - `Truncate` — `Restart` plus truncate the `-wal` file to zero bytes on success.
/// - `Noop` — do not checkpoint any frame; report the current counts only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointMode {
    Passive,
    Full,
    Restart,
    Truncate,
    Noop,
}

impl CheckpointMode {
    /// Parse a `PRAGMA wal_checkpoint(<name>)` argument (case-insensitive). An absent
    /// argument is the bare `PRAGMA wal_checkpoint`, which pragma.html defines as
    /// PASSIVE; an unrecognized name likewise falls back to PASSIVE (SQLite does not
    /// error on it), so the parser never rejects a checkpoint pragma over its mode.
    pub fn from_pragma_arg(name: Option<&str>) -> CheckpointMode {
        match name.map(str::to_ascii_lowercase).as_deref() {
            Some("full") => CheckpointMode::Full,
            Some("restart") => CheckpointMode::Restart,
            Some("truncate") => CheckpointMode::Truncate,
            Some("noop") => CheckpointMode::Noop,
            // bare, "passive", or anything unrecognized
            _ => CheckpointMode::Passive,
        }
    }
}

/// The outcome of a checkpoint, mapped by the engine to `PRAGMA wal_checkpoint`'s
/// documented one-row `(busy, log, checkpointed)` result (pragma.html).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointReport {
    /// True iff a RESTART/FULL/TRUNCATE checkpoint was BLOCKED from completing its
    /// stronger guarantee — e.g. another connection was actively reading or writing
    /// (pragma.html: the first column "will be 1 if a RESTART or FULL or TRUNCATE
    /// checkpoint was blocked from completing"). PASSIVE and NOOP are never busy. The
    /// engine maps this to the pragma's `busy` column (0/1).
    pub busy: bool,
    /// The number of frames in the WAL log (pragma's `log` column), or `None` outside
    /// WAL mode (the engine then reports -1).
    pub log: Option<u32>,
    /// The number of frames moved back into the database file (pragma's `checkpointed`
    /// column), or `None` outside WAL mode (the engine then reports -1).
    pub checkpointed: Option<u32>,
}

impl CheckpointReport {
    /// The report a non-WAL backing returns: nothing to checkpoint, never busy, and
    /// no frame counts (the pragma reports `-1` for `log`/`checkpointed`).
    pub fn not_wal() -> CheckpointReport {
        CheckpointReport { busy: false, log: None, checkpointed: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pragma_arg_maps_every_documented_mode_case_insensitively() {
        assert_eq!(CheckpointMode::from_pragma_arg(None), CheckpointMode::Passive);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("passive")), CheckpointMode::Passive);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("PASSIVE")), CheckpointMode::Passive);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("Full")), CheckpointMode::Full);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("RESTART")), CheckpointMode::Restart);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("truncate")), CheckpointMode::Truncate);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("NoOp")), CheckpointMode::Noop);
    }

    #[test]
    fn from_pragma_arg_falls_back_to_passive_on_unknown() {
        // SQLite does not error on an unrecognized wal_checkpoint argument; it behaves
        // as the default (PASSIVE), so the parser must never reject the pragma here.
        assert_eq!(CheckpointMode::from_pragma_arg(Some("bogus")), CheckpointMode::Passive);
        assert_eq!(CheckpointMode::from_pragma_arg(Some("")), CheckpointMode::Passive);
    }

    #[test]
    fn not_wal_report_is_not_busy_and_has_no_counts() {
        let r = CheckpointReport::not_wal();
        assert!(!r.busy);
        assert_eq!(r.log, None);
        assert_eq!(r.checkpointed, None);
    }
}
