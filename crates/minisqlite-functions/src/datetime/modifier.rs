//! Modifiers (`lang_datefunc.html#dtmods`): the transforms applied left-to-right to a
//! time-value. This module parses a modifier string into a [`Mod`] and applies the
//! whole list to a [`DateTime`]: a transforming modifier normalizes it to an [`Instant`]
//! (SQLite's Julian-day compute) first, producing the final value plus the `subsec` flag.
//!
//! Two subtleties drive the shape here:
//!
//! * The `unixepoch`/`julianday`/`auto` modifiers only reinterpret a *numeric*
//!   time-value, and only when they are the first modifier. That resolution happens
//!   in [`apply`] before the main loop, from the pre-parsed modifier list.
//! * `ceiling`/`floor` resolve the day-of-month overflow of the *immediately
//!   preceding* month/year shift. We carry that overflow forward one step: each
//!   modifier captures and clears the pending carry, a month/year shift sets a fresh
//!   carry, `floor` subtracts it, `ceiling` discards it, and any other modifier drops
//!   it — exactly "the next modifier after a time shift".

use minisqlite_types::Value;

use super::compute::{text_of, Computed};
use super::julian::{days_in_month, Instant};
use super::parse::{parse_time_fields, read_ndigits, TimeValue};
use super::value::DateTime;

const MS_PER_DAY: i64 = 86_400_000;

/// A time unit for the `NNN <unit>` shift modifiers, carrying its multiplier (ms) and
/// the SQLite magnitude limit past which the modifier is rejected (NULL). The
/// month/year multipliers are the nominal 30-day / 365-day values SQLite uses for the
/// *fractional* remainder; their integer part is a calendar shift, handled separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unit {
    Second,
    Minute,
    Hour,
    Day,
    Month,
    Year,
}

impl Unit {
    /// Milliseconds per unit for a *uniform* shift (days and below) and per
    /// *fractional* month/year remainder.
    fn ms(self) -> f64 {
        match self {
            Unit::Second => 1_000.0,
            Unit::Minute => 60_000.0,
            Unit::Hour => 3_600_000.0,
            Unit::Day => 86_400_000.0,
            Unit::Month => 30.0 * 86_400_000.0,
            Unit::Year => 365.0 * 86_400_000.0,
        }
    }

    /// The maximum `|NNN|` SQLite accepts for this unit; larger yields NULL. The
    /// values bound the reachable date range and, as a side effect, keep the shift
    /// arithmetic well within `i64`.
    fn limit(self) -> f64 {
        match self {
            Unit::Second => 4.6427e14,
            Unit::Minute => 7.7379e12,
            Unit::Hour => 1.2897e11,
            Unit::Day => 5_373_485.0,
            Unit::Month => 176_546.0,
            Unit::Year => 14_713.0,
        }
    }
}

/// A parsed modifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mod {
    /// `NNN days`/`hours`/.../`years` (trailing 's' optional, NNN a signed float).
    Add { amount: f64, unit: Unit },
    /// `±HH:MM[:SS[.fff]]` — a uniform time offset in ms.
    TimeOffset(i64),
    /// `±YYYY-MM-DD[ HH:MM[:SS[.fff]]]` — year, then month, then day+time.
    DateOffset { sign: i64, years: i64, months: i64, days: i64, time_ms: i64 },
    StartOfMonth,
    StartOfYear,
    StartOfDay,
    /// `weekday N` — advance forward to weekday N (0=Sunday).
    Weekday(u32),
    Ceiling,
    Floor,
    UnixEpoch,
    JulianDay,
    Auto,
    LocalTime,
    Utc,
    Subsec,
}

/// Apply a pre-resolved time-value plus its raw modifier `Value`s. Returns `None`
/// (NULL) if any modifier argument is NULL or unparseable, or if a numeric
/// interpretation is out of range. `tv` has already had `Now` resolved by the caller.
///
/// A `subsec` modifier only raises the output resolution; every *other* modifier
/// transforms the value via SQLite's Julian-day compute. With no transforming modifier
/// the base is passed through in raw field form, so an out-of-range day and a literal
/// hour-24 both render verbatim (`date('2020-09-31')` -> `'2020-09-31'`,
/// `datetime('2009-02-28 24:00')` -> `'2009-02-28 24:00:00'`); a single transform (even
/// `'+0 days'`) computes the instant, normalizing both (-> `'2020-10-01'`).
pub(crate) fn apply(tv: TimeValue, mod_args: &[Value]) -> Option<Computed> {
    // Pre-parse every modifier; a NULL argument or an invalid modifier is NULL.
    let mut mods: Vec<Mod> = Vec::with_capacity(mod_args.len());
    for a in mod_args {
        let text = text_of(a)?; // NULL argument -> None
        mods.push(parse_modifier(&text)?);
    }

    // Resolve the base value, consuming a leading numeric-reinterpretation keyword
    // (`unixepoch`/`julianday`/`auto`) when the time-value is a bare number; otherwise
    // a bare number is a Julian day by default. A resolved string keeps its (possibly
    // raw) `DateTime`.
    let (base, start): (DateTime, usize) = match tv {
        TimeValue::Now => unreachable!("Now is resolved before apply"),
        TimeValue::Resolved(dt) => (dt, 0usize),
        TimeValue::RawNumber(r) => match mods.first() {
            Some(Mod::UnixEpoch) => (DateTime::Normalized(Instant::from_unix_seconds(r)), 1),
            Some(Mod::JulianDay) => (DateTime::Normalized(Instant::from_julian_day(r)), 1),
            Some(Mod::Auto) => (DateTime::Normalized(Instant::from_auto(r)?), 1),
            _ => (DateTime::Normalized(Instant::from_julian_day(r)), 0),
        },
    };

    let active = &mods[start..];
    let subsec = active.iter().any(|m| matches!(m, Mod::Subsec));
    let transforms = active.iter().any(|m| !matches!(m, Mod::Subsec));

    // No transform: pass the base through un-normalized (a raw out-of-range day and a
    // literal hour-24 both survive).
    if !transforms {
        return Some(Computed { dt: base, subsec });
    }

    // At least one transform: normalize once, then apply the modifiers in order.
    let mut instant = base.to_instant();
    let mut pending_floor: i64 = 0;
    for m in active {
        // Each modifier consumes (and clears) the day-overflow carry from the one
        // before it; a month/year shift installs a fresh carry below.
        let carry = pending_floor;
        pending_floor = 0;
        match *m {
            Mod::Add { amount, unit } => pending_floor = apply_add(&mut instant, amount, unit),
            Mod::TimeOffset(ms) => instant.jd_ms = instant.jd_ms.saturating_add(ms),
            Mod::DateOffset { sign, years, months, days, time_ms } => {
                pending_floor = apply_date_offset(&mut instant, sign, years, months, days, time_ms)
            }
            Mod::StartOfMonth => start_of(&mut instant, StartOf::Month),
            Mod::StartOfYear => start_of(&mut instant, StartOf::Year),
            Mod::StartOfDay => start_of(&mut instant, StartOf::Day),
            Mod::Weekday(n) => apply_weekday(&mut instant, n),
            Mod::Floor => {
                instant.jd_ms = instant.jd_ms.saturating_sub(carry.saturating_mul(MS_PER_DAY))
            }
            Mod::Ceiling => {} // carry discarded; ceiling is the default resolution
            Mod::Subsec => {}  // resolution flag only; captured above
            // LIMITATION: a deterministic OS timezone database is not available here,
            // so localtime/utc are treated as no-ops (local == UTC). This matches a
            // process running with TZ=UTC; other zones would diverge. Flagged, tested.
            Mod::LocalTime | Mod::Utc => {}
            // `julianday` anywhere but as the first modifier on a numeric time-value
            // is an error (NULL) per the spec; `unixepoch`/`auto` there are undefined,
            // so we no-op them harmlessly.
            Mod::JulianDay => return None,
            Mod::UnixEpoch | Mod::Auto => {}
        }
    }

    Some(Computed { dt: DateTime::Normalized(instant), subsec })
}

/// Add `amount` of `unit`. For day and below this is a uniform ms shift (no carry);
/// for month/year the integer part is a calendar shift (returning the day-of-month
/// overflow as the floor carry) and the fractional part is added as nominal days.
fn apply_add(inst: &mut Instant, amount: f64, unit: Unit) -> i64 {
    match unit {
        Unit::Second | Unit::Minute | Unit::Hour | Unit::Day => {
            inst.jd_ms = inst.jd_ms.saturating_add((amount * unit.ms()).round() as i64);
            0
        }
        Unit::Month | Unit::Year => {
            let n = amount.trunc() as i64;
            let frac = amount - amount.trunc();
            let carry = if unit == Unit::Month {
                shift_months(inst, n)
            } else {
                shift_years(inst, n)
            };
            // The fractional remainder is nominal 30-day / 365-day, added after the
            // calendar shift (as SQLite does), and does not affect the floor carry.
            inst.jd_ms = inst.jd_ms.saturating_add((frac * unit.ms()).round() as i64);
            carry
        }
    }
}

/// Shift by `n` whole months, keeping the day-of-month and time-of-day. Overflowing
/// days roll into the following month (the default "ceiling"); the overflow amount is
/// returned so a following `floor` can clamp back to the last day of the target month.
fn shift_months(inst: &mut Instant, n: i64) -> i64 {
    let b = inst.breakdown();
    let total = b.month as i64 + n;
    let new_year = b.year.saturating_add((total - 1).div_euclid(12));
    let new_month = (total - 1).rem_euclid(12) + 1;
    let overflow = (b.day as i64 - days_in_month(new_year, new_month as u32)).max(0);
    *inst = Instant::from_civil(new_year, new_month, b.day as i64, b.time_ms());
    overflow
}

/// Shift by `n` whole years, keeping month/day/time. Feb 29 in a non-leap target year
/// overflows into March (ceiling); the overflow is returned for `floor`.
fn shift_years(inst: &mut Instant, n: i64) -> i64 {
    let b = inst.breakdown();
    let new_year = b.year.saturating_add(n);
    let overflow = (b.day as i64 - days_in_month(new_year, b.month)).max(0);
    *inst = Instant::from_civil(new_year, b.month as i64, b.day as i64, b.time_ms());
    overflow
}

/// Apply a `±YYYY-MM-DD[ HH:MM:SS.fff]` offset: year, then month (both calendar), then
/// day and time (uniform ms). The floor carry comes from the last calendar field that
/// actually shifts.
fn apply_date_offset(
    inst: &mut Instant,
    sign: i64,
    years: i64,
    months: i64,
    days: i64,
    time_ms: i64,
) -> i64 {
    let carry_y = shift_years(inst, sign * years);
    let carry_m = shift_months(inst, sign * months);
    let delta = days.saturating_mul(MS_PER_DAY).saturating_add(time_ms);
    inst.jd_ms = inst.jd_ms.saturating_add(sign * delta);
    if months != 0 {
        carry_m
    } else {
        carry_y
    }
}

enum StartOf {
    Month,
    Year,
    Day,
}

/// Truncate backwards to the start of the month, year, or day (zeroing the time).
fn start_of(inst: &mut Instant, kind: StartOf) {
    let b = inst.breakdown();
    let (y, m, d) = match kind {
        StartOf::Month => (b.year, b.month as i64, 1),
        StartOf::Year => (b.year, 1, 1),
        StartOf::Day => (b.year, b.month as i64, b.day as i64),
    };
    *inst = Instant::from_civil(y, m, d, 0);
}

/// Advance forward (0..6 days) to the next date whose weekday is `n` (0=Sunday),
/// preserving the time-of-day; a no-op if already on that weekday.
fn apply_weekday(inst: &mut Instant, n: u32) {
    let dow = inst.day_of_week() as i64;
    let delta = (n as i64 - dow).rem_euclid(7);
    inst.jd_ms = inst.jd_ms.saturating_add(delta * MS_PER_DAY);
}

/// Parse one modifier string into a [`Mod`], or `None` if it is not a recognized
/// modifier. Never panics: all scanning is bounds-checked.
pub(crate) fn parse_modifier(s: &str) -> Option<Mod> {
    let t = s.trim();
    let first = *t.as_bytes().first()?;
    if first.is_ascii_digit() || first == b'+' || first == b'-' || first == b'.' {
        parse_numeric_modifier(t)
    } else {
        parse_keyword_modifier(t)
    }
}

/// Classify a numeric-leading modifier as a time offset (`:` after the hour digits),
/// a date offset (`-` after a signed year), or a `NNN <unit>` shift.
fn parse_numeric_modifier(t: &str) -> Option<Mod> {
    let b = t.as_bytes();
    let mut i = 0;
    let signed = matches!(b.first(), Some(b'+') | Some(b'-'));
    let neg = b.first() == Some(&b'-');
    if signed {
        i += 1;
    }
    while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
        i += 1;
    }
    match b.get(i) {
        Some(b':') => parse_time_offset(t, neg),
        // A date offset requires an explicit sign (spec: formats 10-13).
        Some(b'-') if signed => parse_date_offset(t, neg),
        _ => parse_amount_unit(t),
    }
}

/// `±HH:MM[:SS[.fff]]` as a uniform ms offset (sign optional).
fn parse_time_offset(t: &str, neg: bool) -> Option<Mod> {
    let b = t.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'+') | Some(b'-')) {
        i += 1;
    }
    let ms = parse_time_fields(b, &mut i)?;
    if i != b.len() {
        return None;
    }
    Some(Mod::TimeOffset(if neg { -ms } else { ms }))
}

/// `±YYYY-MM-DD[ /T HH:MM[:SS[.fff]]]` offset. Month/day are deltas, not validated
/// against a calendar, so a two-digit field can exceed 12/31 and is normalized when
/// applied.
fn parse_date_offset(t: &str, neg: bool) -> Option<Mod> {
    let b = t.as_bytes();
    let mut i = 1; // skip the required sign
    let years = read_ndigits(b, &mut i, 4)? as i64;
    if b.get(i) != Some(&b'-') {
        return None;
    }
    i += 1;
    let months = read_ndigits(b, &mut i, 2)? as i64;
    if b.get(i) != Some(&b'-') {
        return None;
    }
    i += 1;
    let days = read_ndigits(b, &mut i, 2)? as i64;

    let mut time_ms = 0i64;
    // Same separator policy as the ISO time-value parser: a space or uppercase 'T' only.
    // A lowercase 't' is not accepted (it makes the whole modifier NULL).
    if matches!(b.get(i), Some(b' ') | Some(b'T')) {
        i += 1;
        time_ms = parse_time_fields(b, &mut i)?;
    }
    if i != b.len() {
        return None;
    }
    let sign = if neg { -1 } else { 1 };
    Some(Mod::DateOffset { sign, years, months, days, time_ms })
}

/// `NNN <unit>` where NNN is a signed float and `<unit>` is one of the six time units
/// (trailing 's' optional). Rejected (NULL) if `|NNN|` exceeds the unit's limit.
fn parse_amount_unit(t: &str) -> Option<Mod> {
    let (amount, consumed) = parse_float_prefix(t)?;
    let rest = t.get(consumed..)?.trim();
    let unit = match rest.to_ascii_lowercase().as_str() {
        "day" | "days" => Unit::Day,
        "hour" | "hours" => Unit::Hour,
        "minute" | "minutes" => Unit::Minute,
        "second" | "seconds" => Unit::Second,
        "month" | "months" => Unit::Month,
        "year" | "years" => Unit::Year,
        _ => return None,
    };
    if amount.abs() > unit.limit() {
        return None;
    }
    Some(Mod::Add { amount, unit })
}

/// Parse a leading float (`[+-]?digits[.digits][e[+-]digits]`) and return its value
/// and the number of bytes consumed, or `None` if no mantissa digit is present.
fn parse_float_prefix(t: &str) -> Option<(f64, usize)> {
    let b = t.as_bytes();
    let mut i = 0;
    if matches!(b.get(i), Some(b'+') | Some(b'-')) {
        i += 1;
    }
    let mut saw_digit = false;
    while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
        i += 1;
        saw_digit = true;
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
            saw_digit = true;
        }
    }
    if !saw_digit {
        return None;
    }
    if matches!(b.get(i), Some(b'e') | Some(b'E')) {
        let mut j = i + 1;
        if matches!(b.get(j), Some(b'+') | Some(b'-')) {
            j += 1;
        }
        let mut saw_exp = false;
        while matches!(b.get(j), Some(c) if c.is_ascii_digit()) {
            j += 1;
            saw_exp = true;
        }
        if saw_exp {
            i = j;
        }
    }
    let val: f64 = t.get(..i)?.parse().ok()?;
    Some((val, i))
}

/// Parse a keyword modifier (case-insensitive), including `weekday N`.
fn parse_keyword_modifier(t: &str) -> Option<Mod> {
    let lower = t.to_ascii_lowercase();
    match lower.as_str() {
        "ceiling" => Some(Mod::Ceiling),
        "floor" => Some(Mod::Floor),
        "unixepoch" => Some(Mod::UnixEpoch),
        "julianday" => Some(Mod::JulianDay),
        "auto" => Some(Mod::Auto),
        "localtime" => Some(Mod::LocalTime),
        "utc" => Some(Mod::Utc),
        "subsec" | "subsecond" => Some(Mod::Subsec),
        "start of month" => Some(Mod::StartOfMonth),
        "start of year" => Some(Mod::StartOfYear),
        "start of day" => Some(Mod::StartOfDay),
        _ => {
            let rest = lower.strip_prefix("weekday")?.trim();
            let n: f64 = rest.parse().ok()?;
            if (0.0..7.0).contains(&n) && n.fract() == 0.0 {
                Some(Mod::Weekday(n as u32))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Mod {
        parse_modifier(s).unwrap_or_else(|| panic!("failed to parse modifier {s:?}"))
    }

    #[test]
    fn parse_amount_units() {
        assert_eq!(parse("+1 day"), Mod::Add { amount: 1.0, unit: Unit::Day });
        assert_eq!(parse("-3 days"), Mod::Add { amount: -3.0, unit: Unit::Day });
        assert_eq!(parse("1.5 hours"), Mod::Add { amount: 1.5, unit: Unit::Hour });
        assert_eq!(parse("2 months"), Mod::Add { amount: 2.0, unit: Unit::Month });
        assert_eq!(parse("+10 YEARS"), Mod::Add { amount: 10.0, unit: Unit::Year });
        assert_eq!(parse(".5 seconds"), Mod::Add { amount: 0.5, unit: Unit::Second });
    }

    #[test]
    fn parse_offsets() {
        assert_eq!(parse("+05:00"), Mod::TimeOffset(5 * 3_600_000));
        assert_eq!(parse("-04:30"), Mod::TimeOffset(-(4 * 3_600_000 + 30 * 60_000)));
        assert_eq!(
            parse("+0001-00-00"),
            Mod::DateOffset { sign: 1, years: 1, months: 0, days: 0, time_ms: 0 }
        );
        assert_eq!(
            parse("-0000-01-00 00:00:00.000"),
            Mod::DateOffset { sign: -1, years: 0, months: 1, days: 0, time_ms: 0 }
        );
        // The date-offset time separator matches the ISO time-value parser: a space or
        // an uppercase 'T' is accepted, a lowercase 't' makes the whole modifier NULL.
        assert_eq!(
            parse("+0000-00-01T12:00"),
            Mod::DateOffset { sign: 1, years: 0, months: 0, days: 1, time_ms: 43_200_000 }
        );
        assert!(parse_modifier("+0000-00-01t12:00").is_none());
    }

    #[test]
    fn parse_keywords() {
        assert_eq!(parse("start of month"), Mod::StartOfMonth);
        assert_eq!(parse("START OF YEAR"), Mod::StartOfYear);
        assert_eq!(parse("weekday 0"), Mod::Weekday(0));
        assert_eq!(parse("weekday 6"), Mod::Weekday(6));
        assert_eq!(parse("Subsec"), Mod::Subsec);
        assert_eq!(parse("subsecond"), Mod::Subsec);
        assert_eq!(parse("unixepoch"), Mod::UnixEpoch);
        assert_eq!(parse("utc"), Mod::Utc);
    }

    #[test]
    fn invalid_modifiers_are_none() {
        for s in ["", "5", "+5", "weekday 7", "weekday", "start of week", "1 fortnight", "0000-01-00"] {
            assert!(parse_modifier(s).is_none(), "{s:?} should be an invalid modifier");
        }
    }

    #[test]
    fn magnitude_limit_rejects() {
        assert!(parse_modifier("100000 years").is_none()); // > 14713
        assert!(parse_modifier("14713 years").is_some());
        assert!(parse_modifier("999999 months").is_none()); // > 176546
    }

    fn day(s: &str, mods: &[&str]) -> (i64, u32, u32) {
        let vals: Vec<Value> = mods.iter().map(|m| Value::Text((*m).into())).collect();
        let tv = super::super::parse::parse_time_value(&Value::Text(s.into())).expect("base date");
        let c = apply(tv, &vals).expect("apply");
        let b = c.dt.civil();
        (b.year, b.month, b.day)
    }

    #[test]
    fn add_days_and_month_overflow() {
        assert_eq!(day("2009-02-13", &["+1 day"]), (2009, 2, 14));
        // The classic month-overflow case: Jan 31 + 1 month rolls Feb 31 -> Mar 3.
        assert_eq!(day("2009-01-31", &["+1 month"]), (2009, 3, 3));
        // Ceiling (default) vs floor for a one-year shift off a leap day.
        assert_eq!(day("2024-02-29", &["+1 year"]), (2025, 3, 1));
        assert_eq!(day("2024-02-29", &["+1 year", "floor"]), (2025, 2, 28));
        assert_eq!(day("2024-02-29", &["+1 year", "ceiling"]), (2025, 3, 1));
        // Two months after Dec 31 2023: ceiling -> Mar 2 (leap), floor -> Feb 29.
        assert_eq!(day("2023-12-31", &["+2 months"]), (2024, 3, 2));
        assert_eq!(day("2023-12-31", &["+2 months", "floor"]), (2024, 2, 29));
    }

    #[test]
    fn start_of_and_weekday() {
        assert_eq!(day("2009-02-13", &["start of month"]), (2009, 2, 1));
        assert_eq!(day("2009-02-13", &["start of year"]), (2009, 1, 1));
        // First Tuesday in October 2009 via the spec's idiom.
        assert_eq!(day("2009-06-15", &["start of year", "+9 months", "weekday 2"]), (2009, 10, 6));
        // Already on the weekday: unchanged (2009-02-13 is a Friday=5).
        assert_eq!(day("2009-02-13", &["weekday 5"]), (2009, 2, 13));
    }

    #[test]
    fn negative_and_date_offset() {
        assert_eq!(day("2009-03-15", &["-0000-01-00 00:00:00.000"]), (2009, 2, 15));
        assert_eq!(day("2009-02-13", &["+0001-01-01"]), (2010, 3, 14));
    }

    #[test]
    fn fractional_month_uses_nominal_days() {
        // Reconstructed SQLite behavior: +1 month (calendar) then +0.5*30 = 15 days.
        // 2009-01-15 +1 month -> 2009-02-15, +15 days -> 2009-03-02. Pinned here as
        // our behavior (fractional month/year semantics are not in the spec HTML).
        assert_eq!(day("2009-01-15", &["+1.5 months"]), (2009, 3, 2));
    }
}
