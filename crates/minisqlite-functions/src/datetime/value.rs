//! A parsed date/time value that is *either* the raw civil fields as typed or an
//! already-normalized instant.
//!
//! SQLite stores the parsed civil fields verbatim and only normalizes when the Julian
//! day is computed (a modifier, or a Julian-day-derived output). So `date('2020-09-31')`
//! returns `'2020-09-31'` even though September has 30 days, while
//! `date('2020-09-31','0 seconds')` returns `'2020-10-01'`; likewise a literal hour 24
//! (`%H` is documented `00-24`) survives on a field render. Both facts are forced by the
//! spec's own equivalence `date ≡ strftime('%F', …)` — the field codes `%F`/`%d`/`%H`
//! read the stored fields, and a normalized instant could never yield hour 24. To
//! reproduce that, a [`DateTime::Raw`] carries the raw civil fields (out-of-range day
//! and hour 24 and all) until [`DateTime::to_instant`] is called; the field-rendering
//! functions read the raw fields, while every modifier and every Julian-day-derived
//! output (`%s`/`%J`/`%j`/`%w`/`julianday`/`unixepoch`/`timediff`) normalizes first.

use super::julian::{Breakdown, Instant};

/// Raw civil fields exactly as parsed. `day` may be out of range for the month (e.g.
/// `31` in September) and `time_ms` (milliseconds since midnight) may reach `86_400_000`
/// for a literal `24:00`; both survive verbatim until the Julian day is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawCivil {
    pub(crate) year: i64,
    pub(crate) month: u32,
    pub(crate) day: u32,
    pub(crate) time_ms: i64,
}

/// A parsed time-value that has been parsed but not necessarily normalized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DateTime {
    /// Civil fields straight from an ISO string with no numeric timezone offset — the
    /// fields (an out-of-range day, a literal hour 24) are rendered verbatim by the
    /// field functions and only normalize when an instant is computed.
    Raw(RawCivil),
    /// A normalized instant (from `now`, a Julian/Unix number, a timezone-adjusted
    /// string, or any applied modifier).
    Normalized(Instant),
}

impl DateTime {
    /// A normalized instant, rolling any day/time overflow (an out-of-range day, a
    /// literal hour 24) — SQLite's lazy Julian-day computation.
    pub(crate) fn to_instant(self) -> Instant {
        match self {
            DateTime::Raw(r) => Instant::from_civil(r.year, r.month as i64, r.day as i64, r.time_ms),
            DateTime::Normalized(i) => i,
        }
    }

    /// The civil fields for field-level rendering. `Raw` fields are returned verbatim
    /// (an out-of-range day and a literal hour 24 and all); a `Normalized` instant is
    /// broken down normally.
    pub(crate) fn civil(self) -> Breakdown {
        match self {
            DateTime::Raw(r) => {
                let mut ms = r.time_ms;
                let hour = (ms / 3_600_000) as u32;
                ms %= 3_600_000;
                let minute = (ms / 60_000) as u32;
                ms %= 60_000;
                let second = (ms / 1_000) as u32;
                let millis = (ms % 1_000) as u32;
                Breakdown { year: r.year, month: r.month, day: r.day, hour, minute, second, millis }
            }
            DateTime::Normalized(i) => i.breakdown(),
        }
    }
}
