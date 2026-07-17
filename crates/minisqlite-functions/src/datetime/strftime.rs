//! `strftime(FORMAT, ...)` substitution (`lang_datefunc.html#strftm`). Every code from
//! the spec's table is supported; an unknown or unsupported code (or a trailing bare
//! `%`) makes the whole result NULL, exactly as documented.

use super::compute::Computed;
use super::julian::{day_of_year, iso_week, week_of_year_monday, week_of_year_sunday, Instant};

/// Render `fmt` against the computed value, or `None` (NULL) if `fmt` contains an
/// unsupported substitution.
///
/// Field codes (`%Y`/`%m`/`%d`/`%H`/`%M`/`%S`/`%F`/`%T`/`%R`/`%e`/`%k`/`%l`/`%f`/`%p`/
/// `%P`) read the raw civil fields, so an un-normalized out-of-range day (and a literal
/// hour 24) round-trip. Julian-day-derived codes (`%s`/`%J`/`%j`/`%w`/`%u`/`%U`/`%W`/
/// `%V`/`%G`/`%g`) instead read the instant, which normalizes first — exactly as SQLite
/// computes `%Y`/`%d` from the stored fields but `%s` from the (normalizing) Julian day.
pub(crate) fn strftime(fmt: &str, c: &Computed) -> Option<String> {
    // Raw fields (literal day / hour 24) for the field codes; the instant for JD codes.
    let raw = c.dt.civil();
    let inst = c.dt.to_instant();
    let nb = inst.breakdown();
    let dow = inst.day_of_week(); // 0=Sunday
    let (iso_y, iso_w) = iso_week(nb.year, nb.month, nb.day);
    let b = &raw;

    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        // A '%' must be followed by a recognized code; a trailing '%' is NULL.
        let code = chars.next()?;
        match code {
            'd' => out.push_str(&format!("{:02}", b.day)),
            // %e/%k/%l are SPACE-padded to width 2 (C's `%2d`), not unpadded: a
            // single-digit day/hour gets a leading space (' 3'), two digits do not.
            'e' => out.push_str(&format!("{:2}", b.day)),
            'f' => out.push_str(&format!("{:02}.{:03}", b.second, b.millis)),
            'F' => out.push_str(&format!("{:04}-{:02}-{:02}", b.year, b.month, b.day)),
            'G' => out.push_str(&format!("{iso_y:04}")),
            'g' => out.push_str(&format!("{:02}", iso_y.rem_euclid(100))),
            'H' => out.push_str(&format!("{:02}", b.hour)),
            'I' => out.push_str(&format!("{:02}", hour12(b.hour))),
            'j' => out.push_str(&format!("{:03}", day_of_year(nb.year, nb.month as i64, nb.day as i64))),
            'J' => out.push_str(&format_g(inst.julian_day(), 16)),
            'k' => out.push_str(&format!("{:2}", b.hour)),
            'l' => out.push_str(&format!("{:2}", hour12(b.hour))),
            'm' => out.push_str(&format!("{:02}", b.month)),
            'M' => out.push_str(&format!("{:02}", b.minute)),
            'p' => out.push_str(if b.hour >= 12 { "PM" } else { "AM" }),
            'P' => out.push_str(if b.hour >= 12 { "pm" } else { "am" }),
            'R' => out.push_str(&format!("{:02}:{:02}", b.hour, b.minute)),
            's' => out.push_str(&format_unix_seconds(inst, c.subsec)),
            'S' => out.push_str(&format!("{:02}", b.second)),
            'T' => out.push_str(&format!("{:02}:{:02}:{:02}", b.hour, b.minute, b.second)),
            'U' => out.push_str(&format!("{:02}", week_of_year_sunday(nb.year, nb.month, nb.day))),
            'u' => out.push_str(&format!("{}", if dow == 0 { 7 } else { dow })),
            'V' => out.push_str(&format!("{iso_w:02}")),
            'w' => out.push_str(&format!("{dow}")),
            'W' => out.push_str(&format!("{:02}", week_of_year_monday(nb.year, nb.month, nb.day))),
            'Y' => out.push_str(&format!("{:04}", b.year)),
            '%' => out.push('%'),
            _ => return None,
        }
    }
    Some(out)
}

/// 12-hour clock hour (1..12) from a 0..23 hour.
fn hour12(hour: u32) -> u32 {
    let h = hour % 12;
    if h == 0 {
        12
    } else {
        h
    }
}

/// `%s`: integer seconds since 1970, or fractional seconds (3 decimals) with `subsec`.
fn format_unix_seconds(inst: Instant, subsec: bool) -> String {
    if subsec {
        format!("{:.3}", inst.unix_millis() as f64 / 1000.0)
    } else {
        format!("{}", inst.unix_seconds_floor())
    }
}

/// A minimal `printf("%.*g")` for `%J` (the fractional Julian day): `sig` significant
/// digits, trailing zeros stripped, fixed notation for exponents in `-4..sig` and
/// scientific otherwise. Julian-day values render as e.g. `2451544.5`.
fn format_g(v: f64, sig: usize) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".to_string()
        } else if v < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    let neg = v < 0.0;
    let a = v.abs();
    // `{:.*e}` gives `sig` significant digits, correctly rounded.
    let s = format!("{:.*e}", sig - 1, a);
    let (mant, exp_str) = s.split_once('e').expect("{:e} always contains 'e'");
    let exp: i32 = exp_str.parse().expect("{:e} exponent is an integer");
    let mut digits: String = mant.chars().filter(|c| *c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let core = if exp < -4 || exp >= sig as i32 {
        let mantissa = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let (esign, emag) = if exp < 0 { ('-', -exp) } else { ('+', exp) };
        format!("{mantissa}e{esign}{emag:02}")
    } else if exp >= 0 {
        let int_len = (exp + 1) as usize;
        if int_len >= digits.len() {
            format!("{digits}{}", "0".repeat(int_len - digits.len()))
        } else {
            format!("{}.{}", &digits[..int_len], &digits[int_len..])
        }
    } else {
        format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
    };
    if neg {
        format!("-{core}")
    } else {
        core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::value::DateTime;

    fn at(y: i64, mo: i64, d: i64, time_ms: i64, subsec: bool) -> Computed {
        Computed { dt: DateTime::Normalized(Instant::from_civil(y, mo, d, time_ms)), subsec }
    }

    fn fmt(f: &str, c: &Computed) -> String {
        strftime(f, c).unwrap_or_else(|| panic!("strftime({f:?}) returned NULL"))
    }

    #[test]
    fn date_and_time_codes() {
        let c = at(2009, 2, 13, (23 * 3600 + 31 * 60 + 30) * 1000, false);
        assert_eq!(fmt("%Y/%m/%d", &c), "2009/02/13");
        assert_eq!(fmt("%F %T", &c), "2009-02-13 23:31:30");
        assert_eq!(fmt("%H:%M:%S", &c), "23:31:30");
        assert_eq!(fmt("%R", &c), "23:31");
        assert_eq!(fmt("%j", &c), "044"); // 31 + 13
        assert_eq!(fmt("%s", &c), "1234567890");
        assert_eq!(fmt("%w", &c), "5"); // Friday
        assert_eq!(fmt("%u", &c), "5");
        assert_eq!(fmt("100%%", &c), "100%");
    }

    #[test]
    fn space_padded_and_12_hour_codes() {
        let c = at(2009, 2, 3, (5 * 3600 + 7 * 60 + 9) * 1000, false); // 05:07:09
        // %e/%k/%l are SPACE-padded to width 2 (C `%2d`): a single digit gets a leading
        // space, two digits are not padded. %d/%H/%I stay zero-padded.
        assert_eq!(fmt("%e", &c), " 3");
        assert_eq!(fmt("%d", &c), "03");
        assert_eq!(fmt("%k", &c), " 5");
        assert_eq!(fmt("%H", &c), "05");
        assert_eq!(fmt("%I%p", &c), "05AM");
        assert_eq!(fmt("%l%P", &c), " 5am");
        // Afternoon: 13:00 -> 01 PM (%I zero-padded) / ' 1' pm (%l space-padded).
        let c = at(2009, 2, 3, 13 * 3600 * 1000, false);
        assert_eq!(fmt("%I%p", &c), "01PM");
        assert_eq!(fmt("%l%P", &c), " 1pm");
        // Two-digit day/hour: no padding (strftime('%e','2009-01-31') = '31').
        let c = at(2009, 1, 31, 23 * 3600 * 1000, false);
        assert_eq!(fmt("%e", &c), "31");
        assert_eq!(fmt("%k", &c), "23");
    }

    #[test]
    fn fractional_and_subsec() {
        let c = at(2009, 2, 13, 30 * 1000 + 250, false); // 00:00:30.250
        assert_eq!(fmt("%f", &c), "30.250");
        let c = at(2009, 2, 13, 250, true); // 00:00:00.250
        assert_eq!(fmt("%s", &c), format!("{:.3}", Instant::from_civil(2009, 2, 13, 250).unix_millis() as f64 / 1000.0));
    }

    #[test]
    fn iso_and_simple_weeks() {
        // 2005-01-01 is ISO 2004-W53.
        let c = at(2005, 1, 1, 0, false);
        assert_eq!(fmt("%G-W%V", &c), "2004-W53");
        assert_eq!(fmt("%g", &c), "04");
        // Simple week numbers on a Monday (2024-01-01).
        let c = at(2024, 1, 1, 0, false);
        assert_eq!(fmt("%U", &c), "00");
        assert_eq!(fmt("%W", &c), "01");
    }

    #[test]
    fn julian_day_code() {
        let c = at(2000, 1, 1, 0, false);
        assert_eq!(fmt("%J", &c), "2451544.5");
    }

    #[test]
    fn unsupported_code_is_null() {
        let c = at(2009, 2, 13, 0, false);
        assert!(strftime("%Q", &c).is_none());
        assert!(strftime("abc%", &c).is_none()); // trailing bare %
    }

    #[test]
    fn format_g_shapes() {
        assert_eq!(format_g(2451544.5, 16), "2451544.5");
        assert_eq!(format_g(0.0, 16), "0");
        assert_eq!(format_g(100.0, 16), "100");
    }
}
