//! Parsing of *time-values* (`lang_datefunc.html#tmval`): the first argument to a
//! date/time function. A time-value is one of the enumerated ISO-8601 forms, a
//! time-only form (which attaches to the date 2000-01-01), the literal `now`, or a
//! raw Julian-day/Unix number (format 12).
//!
//! Parsing is deliberately strict about structure (fixed digit widths, required
//! separators) and never panics: every byte access is bounds-checked and a
//! malformed value yields `None`, which the caller turns into SQL NULL — SQLite
//! returns NULL, not an error, for an unparseable date string.
//!
//! The shared field scanners (`read_ndigits`, `parse_time_fields`, `parse_tz`) are
//! `pub(crate)` because the modifier parser reuses them for the `±HH:MM` and
//! `±YYYY-MM-DD` offset forms.

use minisqlite_types::{looks_like_integer, looks_like_real, parse_real_prefix, Value};

use super::julian::Instant;
use super::value::{DateTime, RawCivil};

/// A parsed time-value, before the wall clock or a numeric-reinterpretation modifier
/// is applied. `Now` is resolved against the context by the caller; `RawNumber`'s
/// interpretation (Julian day vs Unix) depends on the first modifier. `Resolved`
/// carries a [`DateTime`] that may still be un-normalized — its civil fields are kept
/// verbatim until a modifier or a Julian-day output forces the Julian day.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TimeValue {
    Now,
    Resolved(DateTime),
    RawNumber(f64),
}

/// The trailing timezone of an ISO time-value. `Local` covers both an absent zone and
/// the `Z`/`z` (Zulu) suffix, which the spec calls a no-op — so the civil fields are
/// kept raw (an out-of-range day and a literal hour 24 both survive). A numeric `±HH:MM`
/// offset must be subtracted to reach UTC, which folds the value into a Julian-day
/// instant.
enum Tz {
    Local,
    Offset(i64),
}

/// Parse a time-value from a SQL `Value`. NULL yields `None` (NULL result).
/// INTEGER/REAL are format-12 numbers; TEXT/BLOB are parsed as strings.
pub(crate) fn parse_time_value(v: &Value) -> Option<TimeValue> {
    match v {
        Value::Null => None,
        Value::Integer(i) => Some(TimeValue::RawNumber(*i as f64)),
        Value::Real(r) => Some(TimeValue::RawNumber(*r)),
        Value::Text(s) => parse_time_text(s),
        Value::Blob(b) => parse_time_text(&String::from_utf8_lossy(b)),
    }
}

/// Parse a time-value string: `now`, then the ISO date(-time) forms, then the
/// time-only forms, then a bare numeric (Julian day). Anything else is `None`.
fn parse_time_text(s: &str) -> Option<TimeValue> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("now") {
        return Some(TimeValue::Now);
    }
    if let Some(tv) = parse_iso_datetime(s) {
        return Some(tv);
    }
    if let Some(tv) = parse_iso_time_only(s) {
        return Some(tv);
    }
    // Format 12: the whole string is a Julian-day number (int or real form).
    if looks_like_integer(s) || looks_like_real(s) {
        return Some(TimeValue::RawNumber(parse_real_prefix(s)));
    }
    None
}

/// `YYYY-MM-DD` optionally followed by ` `/`T` + `HH:MM[:SS[.fff]]` and a timezone.
/// The date-time separator is a space or an uppercase `T` only — a lowercase `t` is not
/// accepted (SQLite rejects `date('2009-02-13t00:00')` as NULL). Only the time-bearing
/// forms may carry a timezone (spec: formats 2..10). The parsed civil fields are kept
/// verbatim (an out-of-range day and a literal hour 24 both survive) until a modifier or
/// a Julian-day output normalizes them; a numeric `±HH:MM` offset does force an instant.
fn parse_iso_datetime(s: &str) -> Option<TimeValue> {
    let b = s.as_bytes();
    let mut i = 0;
    let year = read_ndigits(b, &mut i, 4)? as i64;
    expect(b, &mut i, b'-')?;
    let month = read_ndigits(b, &mut i, 2)? as i64;
    expect(b, &mut i, b'-')?;
    let day = read_ndigits(b, &mut i, 2)? as i64;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let mut time_ms = 0i64;
    let mut tz = Tz::Local;
    match b.get(i) {
        None => {} // date only (format 1 — no timezone permitted)
        Some(&sep) if sep == b' ' || sep == b'T' => {
            i += 1;
            time_ms = parse_time_fields(b, &mut i)?;
            tz = parse_tz(b, &mut i)?;
        }
        Some(_) => return None,
    }
    if i != b.len() {
        return None;
    }
    Some(TimeValue::Resolved(resolve_civil(year, month, day, time_ms, tz)))
}

/// `HH:MM[:SS[.fff]]` (+ optional timezone), which attaches to the date 2000-01-01.
fn parse_iso_time_only(s: &str) -> Option<TimeValue> {
    let b = s.as_bytes();
    let mut i = 0;
    let time_ms = parse_time_fields(b, &mut i)?;
    let tz = parse_tz(b, &mut i)?;
    if i != b.len() {
        return None;
    }
    Some(TimeValue::Resolved(resolve_civil(2000, 1, 1, time_ms, tz)))
}

/// Build a [`DateTime`] from parsed civil fields. SQLite stores the parsed fields
/// *verbatim* and only normalizes when the Julian day is computed (a modifier, or a
/// Julian-day-derived output). So an in-range-but-impossible day (`01`-`31` beyond the
/// month's length, e.g. Feb 31) and a literal hour 24 (`%H` is documented `00-24`) both
/// render unchanged from the field functions and only fold over via
/// [`DateTime::to_instant`]. This is why `date('2009-02-31')` = `'2009-02-31'` but
/// `date('2009-02-31','+0 days')` = `'2009-03-03'`, and it is forced by the spec: `date`
/// is *exactly* `strftime('%F', …)` and `%F`/`%d`/`%H` all read the same stored fields
/// (a normalized instant could never yield the documented hour 24).
///
/// A `Local`/Zulu zone keeps the fields raw (a [`DateTime::Raw`]); a numeric `±HH:MM`
/// offset must be subtracted to reach UTC (`UTC = local - offset`), which needs the
/// Julian day and so yields a normalized instant.
fn resolve_civil(year: i64, month: i64, day: i64, time_ms: i64, tz: Tz) -> DateTime {
    match tz {
        Tz::Local => DateTime::Raw(RawCivil {
            year,
            month: month as u32,
            day: day as u32,
            time_ms,
        }),
        Tz::Offset(off) => DateTime::Normalized(Instant::from_civil(year, month, day, time_ms - off)),
    }
}

/// Read exactly `n` ASCII digits at `*i` as an unsigned value, advancing `*i`.
/// Returns `None` (without a partial advance guarantee beyond what was read) if
/// fewer than `n` digits are present.
pub(crate) fn read_ndigits(b: &[u8], i: &mut usize, n: usize) -> Option<u64> {
    let mut v = 0u64;
    for _ in 0..n {
        let c = *b.get(*i)?;
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.wrapping_mul(10).wrapping_add((c - b'0') as u64);
        *i += 1;
    }
    Some(v)
}

/// Consume the single byte `c` at `*i`, or return `None`.
fn expect(b: &[u8], i: &mut usize, c: u8) -> Option<()> {
    if b.get(*i) == Some(&c) {
        *i += 1;
        Some(())
    } else {
        None
    }
}

/// Parse `HH:MM[:SS[.fff]]` at `*i`, returning milliseconds since midnight. Validates
/// `HH<=24`, `MM<=59`, `SS<=59` (out of range -> `None`). An `HH` of 24 is accepted
/// and rolls into the next day when assembled. Fractional seconds may have any number
/// of digits; the result is rounded to the millisecond (so `.9999` rounds up to a
/// full second, which carries when assembled).
pub(crate) fn parse_time_fields(b: &[u8], i: &mut usize) -> Option<i64> {
    let hh = read_ndigits(b, i, 2)? as i64;
    expect(b, i, b':')?;
    let mm = read_ndigits(b, i, 2)? as i64;
    let mut ss = 0i64;
    let mut frac_ms = 0i64;
    if b.get(*i) == Some(&b':') {
        *i += 1;
        ss = read_ndigits(b, i, 2)? as i64;
        if b.get(*i) == Some(&b'.') {
            *i += 1;
            frac_ms = read_fraction_ms(b, i)?;
        }
    }
    if hh > 24 || mm > 59 || ss > 59 {
        return None;
    }
    Some((hh * 3600 + mm * 60 + ss) * 1000 + frac_ms)
}

/// Read one or more fractional-second digits at `*i` and return the value rounded to
/// milliseconds. Requires at least one digit.
fn read_fraction_ms(b: &[u8], i: &mut usize) -> Option<i64> {
    let start = *i;
    while matches!(b.get(*i), Some(c) if c.is_ascii_digit()) {
        *i += 1;
    }
    if *i == start {
        return None;
    }
    // Parse the exact fractional value and round to ms, matching SQLite's
    // seconds-as-double then +0.5 truncation (e.g. ".9999" -> 1000ms).
    let digits = core::str::from_utf8(&b[start..*i]).ok()?;
    let frac: f64 = format!("0.{digits}").parse().ok()?;
    Some((frac * 1000.0).round() as i64)
}

/// Parse an optional trailing timezone at `*i`: `Local` for none or `Z`/`z` (a no-op
/// per the spec, so the civil fields stay raw), or `Offset(ms)` for a numeric `±HH:MM`.
/// A non-timezone byte is left for the caller's full-consumption check (so trailing
/// garbage fails the overall parse).
fn parse_tz(b: &[u8], i: &mut usize) -> Option<Tz> {
    match b.get(*i) {
        None => Some(Tz::Local),
        Some(&c) if c == b'Z' || c == b'z' => {
            *i += 1;
            Some(Tz::Local)
        }
        Some(&c) if c == b'+' || c == b'-' => {
            *i += 1;
            let neg = c == b'-';
            let hh = read_ndigits(b, i, 2)? as i64;
            expect(b, i, b':')?;
            let mm = read_ndigits(b, i, 2)? as i64;
            if hh > 23 || mm > 59 {
                return None;
            }
            let off = (hh * 3600 + mm * 60) * 1000;
            Some(Tz::Offset(if neg { -off } else { off }))
        }
        Some(_) => Some(Tz::Local),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(s: &str) -> DateTime {
        match parse_time_text(s) {
            Some(TimeValue::Resolved(dt)) => dt,
            other => panic!("expected Resolved for {s:?}, got {other:?}"),
        }
    }

    fn ymdhms(s: &str) -> (i64, u32, u32, u32, u32, u32) {
        let b = resolved(s).civil();
        (b.year, b.month, b.day, b.hour, b.minute, b.second)
    }

    #[test]
    fn date_only() {
        assert_eq!(ymdhms("2009-02-13"), (2009, 2, 13, 0, 0, 0));
    }

    #[test]
    fn datetime_forms() {
        assert_eq!(ymdhms("2009-02-13 23:31:30"), (2009, 2, 13, 23, 31, 30));
        assert_eq!(ymdhms("2009-02-13T23:31:30"), (2009, 2, 13, 23, 31, 30));
        assert_eq!(ymdhms("2009-02-13 23:31"), (2009, 2, 13, 23, 31, 0));
    }

    #[test]
    fn fractional_seconds_are_milliseconds() {
        let b = resolved("2009-02-13 23:31:30.250").civil();
        assert_eq!(b.millis, 250);
        // ".5" -> 500ms; more than three digits round to ms.
        assert_eq!(resolved("2009-02-13 00:00:00.5").civil().millis, 500);
        assert_eq!(resolved("2009-02-13 00:00:00.123456").civil().millis, 123);
    }

    #[test]
    fn out_of_range_day_stays_raw_until_normalized() {
        // SQLite keeps an impossible day-of-month verbatim (no modifier): the field
        // functions render the stored day. date('2009-02-31') = '2009-02-31'. (Forced by
        // the spec: date ≡ strftime('%F'), and %d/%H read the same stored fields — a
        // normalized instant could never render the documented %H hour 24.)
        assert!(matches!(resolved("2020-09-31"), DateTime::Raw(_)));
        assert_eq!(ymdhms("2020-09-31"), (2020, 9, 31, 0, 0, 0));
        assert_eq!(ymdhms("2009-02-31"), (2009, 2, 31, 0, 0, 0));
        assert_eq!(ymdhms("2013-02-29"), (2013, 2, 29, 0, 0, 0));
        assert_eq!(ymdhms("2009-02-31 12:34:56"), (2009, 2, 31, 12, 34, 56));
        // A Julian-day computation (any modifier, or a JD output) normalizes: Feb 31 2009
        // rolls to Mar 3.
        let b = resolved("2009-02-31").to_instant().breakdown();
        assert_eq!((b.year, b.month, b.day), (2009, 3, 3));
        // A numeric timezone offset forces the instant already at parse time.
        let tz = resolved("2020-09-31 00:00:00+00:00");
        assert!(matches!(tz, DateTime::Normalized(_)));
        assert_eq!(tz.civil().day, 1); // 2020-10-01
    }

    #[test]
    fn lowercase_t_separator_is_rejected() {
        // SQLite accepts a space or an uppercase 'T' between date and time, but a
        // lowercase 't' is not a valid separator: date('2009-02-13t00:00') is NULL.
        assert!(parse_time_text("2009-02-13t00:00").is_none());
        // The uppercase 'T' and lowercase 'z' zone still parse.
        assert_eq!(ymdhms("2009-02-13T00:00"), (2009, 2, 13, 0, 0, 0));
        assert_eq!(ymdhms("2009-02-13T00:00:00z"), (2009, 2, 13, 0, 0, 0));
    }

    #[test]
    fn time_only_attaches_to_2000_01_01() {
        assert_eq!(ymdhms("14:30"), (2000, 1, 1, 14, 30, 0));
        assert_eq!(ymdhms("14:30:15"), (2000, 1, 1, 14, 30, 15));
    }

    #[test]
    fn timezone_is_subtracted_to_utc() {
        // The spec's equivalence set: all of these are 2013-10-07 08:23:19.120 UTC.
        // Compared as instants because the offset forms normalize while the bare/Z
        // forms stay raw (they are equal as instants, not as `DateTime` variants).
        let utc = resolved("2013-10-07 08:23:19.120").to_instant();
        assert_eq!(resolved("2013-10-07T08:23:19.120Z").to_instant(), utc);
        assert_eq!(resolved("2013-10-07 04:23:19.120-04:00").to_instant(), utc);
        assert_eq!(resolved("2013-10-07 12:23:19.120+04:00").to_instant(), utc);
    }

    #[test]
    fn hour_24_is_kept_raw_and_does_not_roll_the_date() {
        // `%H` is documented as 00-24: a literal hour 24 is preserved verbatim on a field
        // render and does not roll the date — the same stored-field mechanism that keeps
        // an out-of-range day literal. So datetime('2009-02-28 24:00') = '2009-02-28
        // 24:00:00', and both fields stay literal together: '2009-02-31 24:00' keeps day
        // 31 and hour 24.
        assert_eq!(ymdhms("2009-02-28 24:00"), (2009, 2, 28, 24, 0, 0));
        assert_eq!(ymdhms("2009-02-31 24:00"), (2009, 2, 31, 24, 0, 0));
        // A Julian-day computation (as any modifier triggers) folds both the day overflow
        // and the 24:00 into the timeline — the field render is the only place they stay
        // literal.
        let b = resolved("2009-02-28 24:00").to_instant().breakdown();
        assert_eq!((b.year, b.month, b.day, b.hour), (2009, 3, 1, 0));
    }

    #[test]
    fn now_and_numeric() {
        assert_eq!(parse_time_text("now"), Some(TimeValue::Now));
        assert_eq!(parse_time_text("NOW"), Some(TimeValue::Now));
        assert!(matches!(parse_time_text("2451545.0"), Some(TimeValue::RawNumber(r)) if r == 2451545.0));
        assert!(matches!(parse_time_text("1234567890"), Some(TimeValue::RawNumber(r)) if r == 1234567890.0));
    }

    #[test]
    fn malformed_is_none() {
        for s in ["not a date", "2009-13-01x", "2009/02/13", "", "2009-02", "25:00", "12:60"] {
            assert!(parse_time_text(s).is_none(), "{s:?} should be unparseable");
        }
    }

    #[test]
    fn value_dispatch() {
        assert!(parse_time_value(&Value::Null).is_none());
        assert!(matches!(parse_time_value(&Value::Integer(5)), Some(TimeValue::RawNumber(r)) if r == 5.0));
        assert!(matches!(parse_time_value(&Value::Real(2.5)), Some(TimeValue::RawNumber(r)) if r == 2.5));
    }
}
