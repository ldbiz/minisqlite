//! Per-function rendering of a [`Computed`] instant, and the `timediff` calendar
//! difference. `date`/`time`/`datetime` return TEXT, `julianday` a REAL, `unixepoch`
//! an INTEGER (or REAL with `subsec`); `strftime` lives in its own module.

use minisqlite_types::Value;

use super::compute::Computed;
use super::julian::{Breakdown, Instant};

/// `date(...)` -> `YYYY-MM-DD`. Field render from the civil fields, which are the raw
/// parsed fields until a modifier or Julian-day output normalizes them (so an
/// out-of-range day renders verbatim, e.g. `date('2020-09-31')` -> `'2020-09-31'`).
pub(crate) fn render_date(c: &Computed) -> Value {
    let b = c.dt.civil();
    Value::Text(format!("{:04}-{:02}-{:02}", b.year, b.month, b.day))
}

/// `time(...)` -> `HH:MM:SS`, or `HH:MM:SS.SSS` with the `subsec` modifier.
pub(crate) fn render_time(c: &Computed) -> Value {
    let b = c.dt.civil();
    let s = if c.subsec {
        format!("{:02}:{:02}:{:02}.{:03}", b.hour, b.minute, b.second, b.millis)
    } else {
        format!("{:02}:{:02}:{:02}", b.hour, b.minute, b.second)
    };
    Value::Text(s)
}

/// `datetime(...)` -> `YYYY-MM-DD HH:MM:SS`, or `...SS.SSS` with `subsec`.
pub(crate) fn render_datetime(c: &Computed) -> Value {
    let b = c.dt.civil();
    let base = format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        b.year, b.month, b.day, b.hour, b.minute, b.second
    );
    let s = if c.subsec { format!("{base}.{:03}", b.millis) } else { base };
    Value::Text(s)
}

/// `julianday(...)` -> the REAL Julian day (a Julian-day output, so it normalizes).
pub(crate) fn render_julianday(c: &Computed) -> Value {
    Value::Real(c.dt.to_instant().julian_day())
}

/// `unixepoch(...)` -> INTEGER seconds since 1970, or REAL fractional seconds with
/// `subsec` (a Julian-day-derived output, so it normalizes).
pub(crate) fn render_unixepoch(c: &Computed) -> Value {
    let inst = c.dt.to_instant();
    if c.subsec {
        Value::Real(inst.unix_millis() as f64 / 1000.0)
    } else {
        Value::Integer(inst.unix_seconds_floor())
    }
}

/// `timediff(A, B)` -> the calendar difference to add to B to reach A, formatted
/// `(+|-)YYYY-MM-DD HH:MM:SS.SSS`.
///
/// The result must satisfy the spec invariant `datetime(A) == datetime(B,
/// timediff(A,B))`, so it is computed *in the direction and order it will be
/// re-applied*, always relative to B: take as many whole years, then (from the
/// year-shifted instant) as many whole months, as move B toward A without passing it,
/// using the exact same day-of-month rollover `apply_date_offset` uses; the sub-month
/// remainder is then the raw millisecond gap, rendered as a uniform day/time offset.
/// Because every stage mirrors the modifier that later re-applies it — `shift_years`,
/// then `shift_months` on the result, then a uniform `±(days·86_400_000 + time)` — the
/// rollovers and the remainder are identical on both paths, so re-applying the string
/// to B reproduces A exactly.
pub(crate) fn timediff(a: Instant, b: Instant) -> String {
    let forward = a.jd_ms >= b.jd_ms;
    let sign = if forward { '+' } else { '-' };
    let bb = b.breakdown();
    let ab = a.breakdown();

    let year_gap = (ab.year - bb.year).abs();
    let years = greedy_shift(&bb, a.jd_ms, forward, year_gap, shift_years);
    let after_years = shift_years(&bb, if forward { years } else { -years }).breakdown();

    let month_gap = ((ab.year - after_years.year) * 12 + (ab.month as i64 - after_years.month as i64)).abs();
    let months = greedy_shift(&after_years, a.jd_ms, forward, month_gap, shift_months);
    let after_months = shift_months(&after_years, if forward { months } else { -months });

    // Sub-(year+month) remainder: the exact ms gap, applied uniformly as the modifier
    // will (days then time), so re-application lands back on `a` to the millisecond.
    let mut rem = a.jd_ms.saturating_sub(after_months.jd_ms).saturating_abs();
    let day = rem / 86_400_000;
    rem %= 86_400_000;
    let hour = rem / 3_600_000;
    rem %= 3_600_000;
    let minute = rem / 60_000;
    rem %= 60_000;
    let second = rem / 1_000;
    let millis = rem % 1_000;

    format!("{sign}{years:04}-{months:02}-{day:02} {hour:02}:{minute:02}:{second:02}.{millis:03}")
}

/// Largest `n >= 0` such that `shift(from, ±n)` stays on A's side of the instant line
/// (`<= a` when adding, `>= a` when subtracting). `shift` is monotonic in its signed
/// amount, so the true `n` is found by seeding with the nominal field gap and walking
/// to the boundary — up while the next step is still on-side, then down while the
/// current one has overshot. Both walks are a handful of steps for real dates (years
/// gap is exact; months land within ~a year after the year alignment).
fn greedy_shift(
    from: &Breakdown,
    a_ms: i64,
    forward: bool,
    est: i64,
    shift: fn(&Breakdown, i64) -> Instant,
) -> i64 {
    let on_side = |n: i64| {
        let jd = shift(from, if forward { n } else { -n }).jd_ms;
        if forward {
            jd <= a_ms
        } else {
            jd >= a_ms
        }
    };
    let mut n = est.max(0);
    while on_side(n + 1) {
        n += 1;
    }
    while n > 0 && !on_side(n) {
        n -= 1;
    }
    n
}

/// Shift by `k` whole years, keeping month/day/time with the natural day-of-month
/// rollover (Feb 29 -> Mar 1). Mirrors `modifier::shift_years` so `timediff`'s output
/// re-applies exactly.
fn shift_years(b: &Breakdown, k: i64) -> Instant {
    Instant::from_civil(b.year.saturating_add(k), b.month as i64, b.day as i64, b.time_ms())
}

/// Shift by `k` whole months, keeping day-of-month and time with natural rollover.
/// Mirrors `modifier::shift_months` (`total = month-1 + k`, Euclidean split).
fn shift_months(b: &Breakdown, k: i64) -> Instant {
    let total = (b.month as i64 - 1).saturating_add(k);
    let new_year = b.year.saturating_add(total.div_euclid(12));
    let new_month = total.rem_euclid(12) + 1;
    Instant::from_civil(new_year, new_month, b.day as i64, b.time_ms())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::value::DateTime;

    fn comp(jd_ms: i64, subsec: bool) -> Computed {
        Computed { dt: DateTime::Normalized(Instant { jd_ms }), subsec }
    }

    fn inst(y: i64, mo: i64, d: i64, time_ms: i64) -> Instant {
        Instant::from_civil(y, mo, d, time_ms)
    }

    fn text(v: Value) -> String {
        match v {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn basic_renders() {
        let c = comp(inst(2009, 2, 13, (23 * 3600 + 31 * 60 + 30) * 1000).jd_ms, false);
        assert_eq!(text(render_date(&c)), "2009-02-13");
        assert_eq!(text(render_time(&c)), "23:31:30");
        assert_eq!(text(render_datetime(&c)), "2009-02-13 23:31:30");
        assert!(matches!(render_unixepoch(&c), Value::Integer(1_234_567_890)));
    }

    #[test]
    fn subsec_renders_milliseconds() {
        let c = comp(inst(2009, 2, 13, 250).jd_ms, true); // 00:00:00.250
        assert_eq!(text(render_time(&c)), "00:00:00.250");
        assert_eq!(text(render_datetime(&c)), "2009-02-13 00:00:00.250");
        match render_unixepoch(&c) {
            Value::Real(r) => assert!((r - (inst(2009, 2, 13, 250).unix_millis() as f64 / 1000.0)).abs() < 1e-9),
            other => panic!("expected Real, got {other:?}"),
        }
    }

    #[test]
    fn julianday_constant() {
        let c = comp(inst(2000, 1, 1, 0).jd_ms, false);
        assert!(matches!(render_julianday(&c), Value::Real(r) if r == 2451544.5));
    }

    #[test]
    fn timediff_documented_examples() {
        // Both spans render identically per the spec, despite different day counts.
        let a = inst(2023, 2, 15, 0);
        let b = inst(2023, 3, 15, 0);
        assert_eq!(timediff(a, b), "-0000-01-00 00:00:00.000");
        let a = inst(2023, 3, 15, 0);
        let b = inst(2023, 4, 15, 0);
        assert_eq!(timediff(a, b), "-0000-01-00 00:00:00.000");
    }

    /// Re-apply a `timediff(a,b)` string to `b` through the real modifier machinery.
    fn apply_diff(diff: &str, b: Instant) -> Instant {
        let vals = vec![Value::Text(diff.to_string())];
        let tv = super::super::parse::TimeValue::Resolved(DateTime::Normalized(b));
        let c = super::super::modifier::apply(tv, &vals)
            .unwrap_or_else(|| panic!("apply diff {diff:?}"));
        c.dt.to_instant()
    }

    #[test]
    fn timediff_specific_values() {
        // Rollover-increment edge: B=2023-05-31 12:00 back to A=2023-03-02 00:00. The
        // nominal month gap (2) under-counts because Feb-31 rolls to Mar 3 >= A, so the
        // greedy must climb to 3 months, leaving a sub-day residual.
        let a = inst(2023, 3, 2, 0);
        let b = inst(2023, 5, 31, 12 * 3_600_000);
        let diff = timediff(a, b);
        assert_eq!(apply_diff(&diff, b).breakdown(), a.breakdown(), "diff was {diff:?}");
        // The month-overflow-shaped span: 30 whole days, not "1 month".
        assert_eq!(timediff(inst(2024, 3, 1, 0), inst(2024, 1, 31, 0)), "+0000-00-30 00:00:00.000");
    }

    #[test]
    fn timediff_satisfies_invariant_over_grid() {
        // The property `datetime(A) == datetime(B, timediff(A,B))` must hold for every
        // pair, in both orderings. The grid deliberately hits month-ends, leap days,
        // year boundaries, large gaps, and sub-second times — exactly where the
        // year-then-month rollover asymmetry bites.
        let times = [0i64, 12 * 3_600_000, 23 * 3_600_000 + 59 * 60_000 + 59_000 + 250];
        let dates: &[(i64, i64, i64)] = &[
            (2024, 1, 31),
            (2024, 2, 29), // leap day
            (2023, 2, 28),
            (2024, 3, 1),
            (2023, 3, 2),
            (2023, 5, 31),
            (2000, 1, 1),
            (1999, 12, 31),
            (2009, 2, 13),
            (1809, 2, 12),
            (2100, 6, 15),
        ];
        let mut instants = Vec::new();
        for &(y, m, d) in dates {
            for &t in &times {
                instants.push(inst(y, m, d, t));
            }
        }
        for &a in &instants {
            for &b in &instants {
                let diff = timediff(a, b);
                let got = apply_diff(&diff, b);
                assert_eq!(
                    got.breakdown(),
                    a.breakdown(),
                    "invariant failed: timediff({:?}, {:?}) = {diff:?} re-applied to B gave {:?}",
                    a.breakdown(),
                    b.breakdown(),
                    got.breakdown()
                );
                // The sign must reflect the ordering.
                assert_eq!(diff.starts_with('-'), a.jd_ms < b.jd_ms, "sign wrong for {diff:?}");
            }
        }
    }
}
