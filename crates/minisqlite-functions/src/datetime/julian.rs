//! Calendar core for the date/time functions: the canonical instant and the exact
//! conversions between it, the proleptic-Gregorian civil fields, and the derived
//! quantities (day-of-week, day-of-year, ISO week) that rendering and modifiers need.
//!
//! The canonical representation of an instant is `jd_ms` — the Julian Day number
//! multiplied by 86_400_000 (i.e. milliseconds since the Julian epoch, noon on
//! -4713-11-24). This is exactly SQLite's internal `iJD`, and it is what makes the
//! modifier arithmetic exact: whole-unit shifts are integer millisecond additions,
//! and the millisecond resolution is precisely what the `subsec` modifier exposes.
//!
//! JD->civil reuses the exact, total [`civil_from_unix`] from `minisqlite-expr`; the
//! civil->JD direction is the standard `days_from_civil` inverse (Howard Hinnant),
//! implemented here because that direction is not exported. Everything is saturating
//! on the unbounded paths so no input string can panic.

use minisqlite_expr::civil_from_unix;

/// Julian Day of the Unix epoch (1970-01-01 00:00:00 UTC) is 2440587.5; times
/// 86_400_000 ms/day that is this constant. Subtracting it converts `jd_ms` to Unix
/// milliseconds (and it is a multiple of 1000, so second/millisecond splits of the
/// two representations agree).
pub(crate) const UNIX_EPOCH_JD_MS: i64 = 210_866_760_000_000;

const MS_PER_DAY: i64 = 86_400_000;
const MS_PER_SEC: i64 = 1_000;

/// An instant on the time line, stored as Julian-Day-milliseconds. Copy because it
/// is a single integer that threads through parsing, modifiers, and rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Instant {
    pub(crate) jd_ms: i64,
}

/// The proleptic-Gregorian civil fields of an instant. `year` is a full signed year
/// (0000-9999 is the range SQLite defines; outside it results are undefined but the
/// math stays total). `millis` is the sub-second remainder in `[0, 1000)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Breakdown {
    pub(crate) year: i64,
    pub(crate) month: u32,
    pub(crate) day: u32,
    pub(crate) hour: u32,
    pub(crate) minute: u32,
    pub(crate) second: u32,
    pub(crate) millis: u32,
}

impl Breakdown {
    /// Milliseconds since midnight for this instant's time-of-day, the form the
    /// modifier code reassembles a shifted date with.
    pub(crate) fn time_ms(&self) -> i64 {
        ((self.hour as i64 * 3600) + (self.minute as i64 * 60) + self.second as i64) * MS_PER_SEC
            + self.millis as i64
    }
}

impl Instant {
    /// The instant for the wall-clock 'now', read once from the context (Unix ms).
    pub(crate) fn from_now(now_unix_millis: i64) -> Instant {
        Instant { jd_ms: now_unix_millis.saturating_add(UNIX_EPOCH_JD_MS) }
    }

    /// A raw Julian-day number (time-value format 12, default interpretation).
    /// Rounds to the nearest millisecond, matching SQLite's `r*86400000 + 0.5`.
    pub(crate) fn from_julian_day(r: f64) -> Instant {
        Instant { jd_ms: round_to_i64(r * MS_PER_DAY as f64) }
    }

    /// A Unix timestamp in seconds (the `unixepoch` interpretation of format 12).
    pub(crate) fn from_unix_seconds(r: f64) -> Instant {
        Instant { jd_ms: round_to_i64(r * MS_PER_SEC as f64).saturating_add(UNIX_EPOCH_JD_MS) }
    }

    /// The `auto` modifier's magnitude-based interpretation of a numeric time-value:
    /// a valid Julian day in `[0, 5373484.499999]`, else a Unix timestamp in
    /// `[-210866760000, 253402300799]`, else out of range (NULL). Ranges are from
    /// `lang_datefunc.html#automod`.
    pub(crate) fn from_auto(r: f64) -> Option<Instant> {
        if (0.0..=5_373_484.499_999).contains(&r) {
            Some(Instant::from_julian_day(r))
        } else if (-210_866_760_000.0..=253_402_300_799.0).contains(&r) {
            Some(Instant::from_unix_seconds(r))
        } else {
            None
        }
    }

    /// Build an instant from civil year/month/day plus a milliseconds-since-midnight
    /// offset. `month` must be 1..=12; `day` may exceed the month length and rolls
    /// over naturally (this is exactly the default "ceiling" resolution of a
    /// month/year shift that overflows the day-of-month). `time_ms` need not be in
    /// `[0, 86400000)` — it is added as a plain offset, so a negative timezone
    /// adjustment or a `24:00` hour lands on the right instant.
    pub(crate) fn from_civil(year: i64, month: i64, day: i64, time_ms: i64) -> Instant {
        let days = days_from_civil(year, month, day);
        let unix_ms = days.saturating_mul(MS_PER_DAY).saturating_add(time_ms);
        Instant { jd_ms: unix_ms.saturating_add(UNIX_EPOCH_JD_MS) }
    }

    /// Decompose into civil fields. Total across the whole `i64` range because
    /// [`civil_from_unix`] is and the epoch shift saturates (a saturated `jd_ms` from
    /// an extreme input yields a clamped-but-panic-free breakdown).
    pub(crate) fn breakdown(self) -> Breakdown {
        let unix_ms = self.jd_ms.saturating_sub(UNIX_EPOCH_JD_MS);
        let unix_secs = unix_ms.div_euclid(MS_PER_SEC);
        let millis = unix_ms.rem_euclid(MS_PER_SEC) as u32;
        let (year, month, day, hour, minute, second) = civil_from_unix(unix_secs);
        Breakdown { year, month, day, hour, minute, second, millis }
    }

    /// The Julian day as a fractional real (the `julianday()` result and `%J`).
    pub(crate) fn julian_day(self) -> f64 {
        self.jd_ms as f64 / MS_PER_DAY as f64
    }

    /// Unix milliseconds (may be negative before 1970). Saturating so an extreme
    /// `jd_ms` cannot overflow the subtraction.
    pub(crate) fn unix_millis(self) -> i64 {
        self.jd_ms.saturating_sub(UNIX_EPOCH_JD_MS)
    }

    /// Floored Unix seconds (the integer `unixepoch()` / `%s` result). Floor, not
    /// truncate, so a pre-1970 fractional second reports the earlier whole second.
    pub(crate) fn unix_seconds_floor(self) -> i64 {
        self.unix_millis().div_euclid(MS_PER_SEC)
    }

    /// Day of week with Sunday=0..Saturday=6. Derived from the whole-day index so it
    /// is correct for any instant, including pre-epoch (1970-01-01 was a Thursday=4).
    pub(crate) fn day_of_week(self) -> u32 {
        let days = self.unix_millis().div_euclid(MS_PER_DAY);
        (days + 4).rem_euclid(7) as u32
    }
}

/// Round a finite `f64` to the nearest `i64` (half away from zero, matching SQLite's
/// `+/-0.5`-then-truncate), saturating on overflow and mapping NaN to 0 so no cast
/// can panic or wrap.
fn round_to_i64(x: f64) -> i64 {
    // `f64 as i64` saturates to i64::MIN/MAX for out-of-range and gives 0 for NaN
    // (guaranteed since Rust 1.45), so `.round()` then `as` is total.
    x.round() as i64
}

/// Days from 1970-01-01 to civil `year-month-day`, the exact inverse of the
/// days->civil breakdown. `month` is 1..=12; `day` may be out of the 1..=month-length
/// range and the result rolls over correctly (the algorithm computes a day-of-year
/// offset, so `days_from_civil(2009, 2, 31)` equals `days_from_civil(2009, 3, 3)`).
/// This is Howard Hinnant's `days_from_civil`; it is the inverse of the
/// `civil_from_unix` we reuse, and is verified to round-trip against it in tests.
pub(crate) fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    // Saturating on the products/sums that scale with `year`, so a wildly out-of-range
    // civil year (e.g. i64::MAX from a saturated shift) clamps instead of panicking;
    // for every in-range date (SQLite's 0000..9999 and far beyond) the values are tiny
    // and no clamp is reached, so this stays exact where it matters.
    let y = if month <= 2 { year.saturating_sub(1) } else { year };
    let era = (if y >= 0 { y } else { y.saturating_sub(399) }) / 400;
    let yoe = y.saturating_sub(era.saturating_mul(400)); // [0, 399]
    let mp = if month > 2 { month - 3 } else { month + 9 }; // Mar=0 .. Feb=11
    let doy = (153 * mp + 2) / 5 + day - 1; // day-of-year from Mar 1 (+ overflow)
    let doe = yoe
        .saturating_mul(365)
        .saturating_add(yoe / 4)
        .saturating_sub(yoe / 100)
        .saturating_add(doy);
    era.saturating_mul(146_097).saturating_add(doe).saturating_sub(719_468)
}

/// Whether `year` is a leap year in the proleptic Gregorian calendar.
pub(crate) fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Number of days in civil `year`/`month` (`month` is 1..=12; other values yield 30
/// as a harmless fallback, never reached because callers normalize first).
pub(crate) fn days_in_month(year: i64, month: u32) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// 1-based day of the year (Jan 1 == 1).
pub(crate) fn day_of_year(year: i64, month: i64, day: i64) -> i64 {
    days_from_civil(year, month, day) - days_from_civil(year, 1, 1) + 1
}

/// `%U`: week of year 00..53 where week 01 begins on the first Sunday. Days before
/// the first Sunday are week 00.
pub(crate) fn week_of_year_sunday(year: i64, month: u32, day: u32) -> i64 {
    let yday0 = day_of_year(year, month as i64, day as i64) - 1;
    let dow = weekday_sunday0(year, month, day);
    (yday0 - dow + 7) / 7
}

/// `%W`: week of year 00..53 where week 01 begins on the first Monday.
pub(crate) fn week_of_year_monday(year: i64, month: u32, day: u32) -> i64 {
    let yday0 = day_of_year(year, month as i64, day as i64) - 1;
    let dow_mon0 = (weekday_sunday0(year, month, day) + 6) % 7;
    (yday0 - dow_mon0 + 7) / 7
}

/// Day-of-week (Sunday=0) for a civil date, used by the week-number helpers.
fn weekday_sunday0(year: i64, month: u32, day: u32) -> i64 {
    (days_from_civil(year, month as i64, day as i64) + 4).rem_euclid(7)
}

/// The ISO-8601 week date `(iso_year, iso_week)` for a civil date (`%G`/`%g` and
/// `%V`). ISO weeks start on Monday and week 01 is the week containing the first
/// Thursday (equivalently, containing Jan 4), so early-January and late-December
/// dates can belong to the neighbouring ISO year.
pub(crate) fn iso_week(year: i64, month: u32, day: u32) -> (i64, i64) {
    // Monday=1 .. Sunday=7.
    let dow_mon1 = ((days_from_civil(year, month as i64, day as i64) + 3).rem_euclid(7)) + 1;
    let yday = day_of_year(year, month as i64, day as i64);
    let week = (yday - dow_mon1 + 10) / 7;
    if week < 1 {
        let prev = year - 1;
        (prev, iso_weeks_in_year(prev))
    } else if week > iso_weeks_in_year(year) {
        (year + 1, 1)
    } else {
        (year, week)
    }
}

/// Number of ISO weeks (52 or 53) in `year`. A year has 53 ISO weeks iff Jan 1 is a
/// Thursday, or it is a leap year whose Jan 1 is a Wednesday.
fn iso_weeks_in_year(year: i64) -> i64 {
    let jan1_mon1 = ((days_from_civil(year, 1, 1) + 3).rem_euclid(7)) + 1;
    if jan1_mon1 == 4 || (is_leap(year) && jan1_mon1 == 3) {
        53
    } else {
        52
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_known_instants_round_trip() {
        // Unix epoch.
        let e = Instant::from_unix_seconds(0.0);
        assert_eq!(e.jd_ms, UNIX_EPOCH_JD_MS);
        let b = e.breakdown();
        assert_eq!((b.year, b.month, b.day, b.hour, b.minute, b.second), (1970, 1, 1, 0, 0, 0));
        // The widely-cited 1234567890 == 2009-02-13 23:31:30 UTC.
        let t = Instant::from_unix_seconds(1_234_567_890.0);
        let b = t.breakdown();
        assert_eq!(
            (b.year, b.month, b.day, b.hour, b.minute, b.second),
            (2009, 2, 13, 23, 31, 30)
        );
        assert_eq!(t.unix_seconds_floor(), 1_234_567_890);
    }

    #[test]
    fn julian_day_of_2000_01_01_is_2451544_5() {
        let i = Instant::from_civil(2000, 1, 1, 0);
        assert_eq!(i.julian_day(), 2451544.5);
    }

    #[test]
    fn from_julian_day_round_trips_civil() {
        // JD 2451544.5 is midnight 2000-01-01 (a Julian day's fraction .5 == 00:00
        // UTC, since JD .0 is noon). JD 2451545.0 would be that day's noon.
        let midnight = Instant::from_julian_day(2451544.5).breakdown();
        assert_eq!(
            (midnight.year, midnight.month, midnight.day, midnight.hour, midnight.minute, midnight.second),
            (2000, 1, 1, 0, 0, 0)
        );
        let noon = Instant::from_julian_day(2451545.0).breakdown();
        assert_eq!(
            (noon.year, noon.month, noon.day, noon.hour, noon.minute, noon.second),
            (2000, 1, 1, 12, 0, 0)
        );
    }

    // The exported civil_from_unix and our days_from_civil must be exact inverses.
    // Exhaustively check a dense day range spanning leap years and a century.
    #[test]
    fn days_from_civil_inverts_civil_from_unix() {
        // Every day across ~1904..2096 (spans 1900/2000 leap rules via 4-year cycles).
        let start = days_from_civil(1904, 1, 1);
        let end = days_from_civil(2096, 12, 31);
        let mut d = start;
        while d <= end {
            let secs = d * 86_400;
            let (y, mo, da, _, _, _) = civil_from_unix(secs);
            assert_eq!(days_from_civil(y, mo as i64, da as i64), d, "round trip failed at day {d}");
            d += 1;
        }
    }

    #[test]
    fn day_overflow_rolls_over() {
        // Feb 31 in a non-leap year is March 3.
        assert_eq!(days_from_civil(2009, 2, 31), days_from_civil(2009, 3, 3));
        // Feb 31 in a leap year is March 2.
        assert_eq!(days_from_civil(2024, 2, 31), days_from_civil(2024, 3, 2));
    }

    #[test]
    fn days_in_month_leap_rules() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2000, 2), 29);
        assert_eq!(days_in_month(1900, 2), 28);
        assert_eq!(days_in_month(2024, 4), 30);
        assert_eq!(days_in_month(2024, 1), 31);
    }

    #[test]
    fn weekday_known_values() {
        // 2009-02-13 was a Friday (Sunday=0 => Friday=5).
        assert_eq!(Instant::from_civil(2009, 2, 13, 0).day_of_week(), 5);
        // 1970-01-01 was a Thursday=4.
        assert_eq!(Instant::from_civil(1970, 1, 1, 0).day_of_week(), 4);
        // 2000-01-01 was a Saturday=6.
        assert_eq!(Instant::from_civil(2000, 1, 1, 0).day_of_week(), 6);
    }

    #[test]
    fn day_of_year_values() {
        assert_eq!(day_of_year(2024, 1, 1), 1);
        assert_eq!(day_of_year(2024, 12, 31), 366); // leap
        assert_eq!(day_of_year(2023, 12, 31), 365);
        assert_eq!(day_of_year(2024, 3, 1), 61); // 31 + 29 + 1
    }

    #[test]
    fn iso_week_known_values() {
        // 2005-01-01 (Sat) belongs to ISO week 53 of 2004.
        assert_eq!(iso_week(2005, 1, 1), (2004, 53));
        // 2004-01-01 (Thu) is ISO week 1 of 2004.
        assert_eq!(iso_week(2004, 1, 1), (2004, 1));
        // 2007-12-31 (Mon) is ISO week 1 of 2008.
        assert_eq!(iso_week(2007, 12, 31), (2008, 1));
        // 2023-01-01 (Sun) belongs to ISO week 52 of 2022.
        assert_eq!(iso_week(2023, 1, 1), (2022, 52));
    }

    #[test]
    fn simple_week_numbers() {
        // Jan 1 2023 is a Sunday: %U starts a new week (01), %W is still 00.
        assert_eq!(week_of_year_sunday(2023, 1, 1), 1);
        assert_eq!(week_of_year_monday(2023, 1, 1), 0);
        // Jan 1 2024 is a Monday: %U is 00, %W starts week 01.
        assert_eq!(week_of_year_sunday(2024, 1, 1), 0);
        assert_eq!(week_of_year_monday(2024, 1, 1), 1);
    }

    #[test]
    fn extreme_inputs_do_not_panic() {
        // Saturating paths: enormous julian day and civil year must not panic.
        let _ = Instant::from_julian_day(f64::MAX).breakdown();
        let _ = Instant::from_julian_day(-f64::MAX).breakdown();
        let _ = Instant::from_civil(i64::MAX, 1, 1, 0).breakdown();
        let _ = Instant::from_unix_seconds(f64::NAN);
    }
}
