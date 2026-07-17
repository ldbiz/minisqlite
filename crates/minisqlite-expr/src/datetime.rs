//! Civil (calendar) time from a Unix timestamp, and the `CURRENT_*` renderings.
//!
//! [`civil_from_unix`] is a closed-form, allocation-free, loop-free conversion (so
//! it is exact and bounded for the whole `i64` second range) that the date/time
//! function family will reuse. The algorithm is the standard
//! days-from-civil inverse (Howard Hinnant's `civil_from_days`), which is exact for
//! every day in range and correct across the proleptic Gregorian calendar.

use crate::ir::NowKind;

/// Break a Unix timestamp (seconds since 1970-01-01 UTC) into UTC civil fields
/// `(year, month, day, hour, minute, second)`.
///
/// Uses floored division so negative (pre-1970) timestamps land on the correct
/// civil day and a non-negative time-of-day. `month`/`day` are 1-based; `hour` is
/// `0..24`. `year` is a full proleptic Gregorian year (can be negative for very
/// distant pasts) — for the `CURRENT_*` keywords it is always a normal positive
/// year, but the function is total across the range for the date/time functions.
pub fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    // Floored split into whole days and second-of-day in [0, 86400).
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let hour = (sod / 3_600) as u32;
    let minute = ((sod % 3_600) / 60) as u32;
    let second = (sod % 60) as u32;

    // civil_from_days (Hinnant): shift the epoch to 0000-03-01 so leap days land at
    // the end of the 400-year era, making the arithmetic branch-free.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day-of-era, [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], from March 1
    let mp = (5 * doy + 2) / 153; // month shifted so March=0, [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    let year = y + if month <= 2 { 1 } else { 0 };

    (year, month, day, hour, minute, second)
}

/// Render a `CURRENT_DATE`/`CURRENT_TIME`/`CURRENT_TIMESTAMP` value from a
/// wall-clock reading in Unix milliseconds. The clock is read in the shell (via
/// [`crate::FnContext::now_unix_millis`]); this is the pure formatting half.
pub fn format_now(kind: NowKind, now_unix_millis: i64) -> String {
    let secs = now_unix_millis.div_euclid(1_000);
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    match kind {
        NowKind::Date => format!("{y:04}-{mo:02}-{d:02}"),
        NowKind::Time => format!("{h:02}:{mi:02}:{s:02}"),
        NowKind::Timestamp => format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_1970_01_01() {
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn a_known_timestamp() {
        // 1234567890 == 2009-02-13 23:31:30 UTC (a widely-cited epoch value).
        assert_eq!(civil_from_unix(1_234_567_890), (2009, 2, 13, 23, 31, 30));
    }

    #[test]
    fn day_boundaries_and_time_of_day() {
        assert_eq!(civil_from_unix(86_399), (1970, 1, 1, 23, 59, 59));
        assert_eq!(civil_from_unix(86_400), (1970, 1, 2, 0, 0, 0));
    }

    #[test]
    fn leap_day_2000() {
        // 2000 is a leap year (divisible by 400): Feb 29 exists.
        // 951782400 == 2000-02-29 00:00:00 UTC.
        assert_eq!(civil_from_unix(951_782_400), (2000, 2, 29, 0, 0, 0));
    }

    #[test]
    fn pre_epoch_is_floored() {
        // One second before the epoch is 1969-12-31 23:59:59, not a negative
        // time-of-day.
        assert_eq!(civil_from_unix(-1), (1969, 12, 31, 23, 59, 59));
        assert_eq!(civil_from_unix(-86_400), (1969, 12, 31, 0, 0, 0));
    }

    #[test]
    fn formats_match_sqlite_layout() {
        let ms = 1_234_567_890_000; // 2009-02-13 23:31:30 UTC
        assert_eq!(format_now(NowKind::Date, ms), "2009-02-13");
        assert_eq!(format_now(NowKind::Time, ms), "23:31:30");
        assert_eq!(format_now(NowKind::Timestamp, ms), "2009-02-13 23:31:30");
    }

    // A round-trip against the forward days_from_civil for a spread of dates keeps
    // the closed form honest without importing a date library as an oracle.
    #[test]
    fn round_trips_days_from_civil() {
        fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
            let y = if m <= 2 { y - 1 } else { y };
            let era = if y >= 0 { y } else { y - 399 } / 400;
            let yoe = y - era * 400;
            let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
            let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
            era * 146_097 + doe - 719_468
        }
        for &(y, m, d) in &[
            (1970, 1, 1),
            (1999, 12, 31),
            (2000, 1, 1),
            (2024, 2, 29),
            (2026, 7, 6),
            (1900, 3, 1),
            (1969, 12, 31),
        ] {
            let secs = days_from_civil(y, m, d) * 86_400 + 12 * 3_600 + 34 * 60 + 56;
            assert_eq!(civil_from_unix(secs), (y, m as u32, d as u32, 12, 34, 56));
        }
    }
}
