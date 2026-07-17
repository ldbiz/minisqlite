//! Conformance battery for SQLite's built-in date/time functions.
//!
//! Every expectation here is TRANSCRIBED FROM THE SPEC
//! (`spec/sqlite-doc/lang_datefunc.html`) — never from what the engine happens
//! to return. Section citations (e.g. "§3 Modifiers") point at that document.
//!
//! Each case asserts DOCUMENTED spec behavior; a case that reveals an engine bug is
//! left as a genuine failing assertion rather than weakened to pass.
//!
//! Determinism: all inputs are FIXED literals. `now` / `CURRENT_TIME` and
//! friends are non-reproducible and are deliberately absent — every value below
//! can be recomputed by hand from the spec's calendar rules.
//!
//! Layout: many small `#[test]` functions, grouped by function/feature, so one
//! discrepancy fails exactly one case and the rest keep reporting.

mod conformance;

use conformance::*;

// ---------------------------------------------------------------------------
// date() — returns the date as text "YYYY-MM-DD" (§1, "The date() function
// returns the date as text in this format: YYYY-MM-DD").
// ---------------------------------------------------------------------------

#[test]
fn date_returns_yyyy_mm_dd() {
    // A full datetime input: only the date part is kept.
    eval_eq("date('2004-01-01 12:34:56')", text("2004-01-01"));
}

#[test]
fn date_accepts_leap_day() {
    // 2004 is divisible by 4 and not by 100, so it is a leap year and
    // 2004-02-29 is a real date (Gregorian calendar, §5).
    eval_eq("date('2004-02-29')", text("2004-02-29"));
}

#[test]
fn date_from_date_only_input_is_unchanged() {
    eval_eq("date('2004-07-15')", text("2004-07-15"));
}

// ---------------------------------------------------------------------------
// time() — returns "HH:MM:SS" (§1, "The time() function returns the time as
// text ... HH:MM:SS ...").
// ---------------------------------------------------------------------------

#[test]
fn time_returns_hh_mm_ss() {
    eval_eq("time('2004-01-01 12:34:56')", text("12:34:56"));
}

#[test]
fn time_from_date_only_input_is_midnight() {
    // No time component in the input ⇒ 00:00:00.
    eval_eq("time('2004-01-01')", text("00:00:00"));
}

#[test]
fn time_only_input_assumes_date_2000_01_01_but_time_prints_as_given() {
    // §2: "Formats 8 through 10 that specify only a time assume a date of
    // 2000-01-01." time() prints only the time part, so the assumed date is
    // not visible here, but a missing seconds field defaults to :00.
    eval_eq("time('12:34:56')", text("12:34:56"));
    eval_eq("time('12:34')", text("12:34:00"));
}

// ---------------------------------------------------------------------------
// datetime() — returns "YYYY-MM-DD HH:MM:SS" (§1).
// ---------------------------------------------------------------------------

#[test]
fn datetime_returns_yyyy_mm_dd_hh_mm_ss() {
    eval_eq("datetime('2004-01-01 12:34:56')", text("2004-01-01 12:34:56"));
}

#[test]
fn datetime_from_date_only_pads_midnight() {
    eval_eq("datetime('2004-01-01')", text("2004-01-01 00:00:00"));
}

#[test]
fn datetime_t_separator_is_normalized_to_space() {
    // §2 format 6: "YYYY-MM-DDTHH:MM:SS" — the literal 'T' separates date and
    // time on input; output always uses a space.
    eval_eq("datetime('2004-01-01T12:34:56')", text("2004-01-01 12:34:56"));
    eval_eq("date('2004-01-01T12:34:56')", text("2004-01-01"));
    eval_eq("time('2004-01-01T12:34:56')", text("12:34:56"));
}

#[test]
fn datetime_hh_mm_input_defaults_seconds_to_zero() {
    // §2 format 2: "YYYY-MM-DD HH:MM" — a missing seconds field is :00.
    eval_eq("datetime('2004-01-01 12:34')", text("2004-01-01 12:34:00"));
}

#[test]
fn datetime_time_only_input_assumes_year_2000_01_01() {
    // §2: formats 8–10 (time only) assume the date 2000-01-01.
    eval_eq("datetime('12:34:56')", text("2000-01-01 12:34:56"));
    eval_eq("datetime('12:34')", text("2000-01-01 12:34:00"));
}

// ---------------------------------------------------------------------------
// Timezone indicators (§2): "Z" is a no-op; a "[+-]HH:MM" suffix is SUBTRACTED
// from the indicated time to obtain zulu (UTC) time. The spec gives these three
// as equivalent to "2013-10-07 08:23:19.120":
//     2013-10-07T08:23:19.120Z
//     2013-10-07 04:23:19.120-04:00
// datetime() (without subsec) prints whole seconds.
// ---------------------------------------------------------------------------

#[test]
fn timezone_z_suffix_is_a_noop() {
    eval_eq("datetime('2013-10-07 08:23:19Z')", text("2013-10-07 08:23:19"));
}

#[test]
fn timezone_offset_is_subtracted_to_reach_utc() {
    // Spec's own equivalence: 04:23:19 with a -04:00 suffix is 08:23:19 UTC,
    // because 04:23:19 - (-04:00) = 08:23:19.
    eval_eq("datetime('2013-10-07 04:23:19-04:00')", text("2013-10-07 08:23:19"));
    // A positive offset subtracts: 12:00 with +02:00 ⇒ 10:00 UTC.
    eval_eq("datetime('2004-01-01 12:00:00+02:00')", text("2004-01-01 10:00:00"));
    // Subtracting a negative offset adds: 12:00 with -05:00 ⇒ 17:00 UTC.
    eval_eq("datetime('2004-01-01 12:00:00-05:00')", text("2004-01-01 17:00:00"));
}

// ---------------------------------------------------------------------------
// Julian-day-number numeric input (§2 format 12). A numeric time-value with no
// 'auto'/'unixepoch' modifier is a Julian day number. JD 2451545.0 is exactly
// 2000-01-01 12:00:00 UTC (the J2000 epoch); JD 2440587.5 is 1970-01-01
// 00:00:00 UTC.
// ---------------------------------------------------------------------------

#[test]
fn numeric_input_is_a_julian_day_number() {
    eval_eq("date(2451545.0)", text("2000-01-01"));
    eval_eq("datetime(2451545.0)", text("2000-01-01 12:00:00"));
    eval_eq("datetime(2440587.5)", text("1970-01-01 00:00:00"));
}

#[test]
fn julianday_roundtrips_a_julian_day_number() {
    // julianday() of a JD number returns that same number.
    assert_scalar_approx(&mut mem(), "SELECT julianday(2451545.0)", 2451545.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Modifiers (§3). Each modifier transforms the value to its left; applied left
// to right.
// ---------------------------------------------------------------------------

#[test]
fn modifier_plus_one_day() {
    eval_eq("date('2004-01-01','+1 day')", text("2004-01-02"));
}

#[test]
fn modifier_day_rollover_across_month_and_year() {
    // Adding a day crosses month and year boundaries by the calendar.
    eval_eq("date('2004-01-31','+1 day')", text("2004-02-01"));
    eval_eq("date('2004-12-31','+1 day')", text("2005-01-01"));
}

#[test]
fn modifier_minus_one_day_into_leap_day() {
    // 2004 is a leap year: the day before 2004-03-01 is 2004-02-29.
    eval_eq("date('2004-03-01','-1 day')", text("2004-02-29"));
}

#[test]
fn modifier_minus_one_day_non_leap_year() {
    // 2003 is not a leap year: the day before 2003-03-01 is 2003-02-28.
    eval_eq("date('2003-03-01','-1 day')", text("2003-02-28"));
}

#[test]
fn modifier_plus_one_month_and_year() {
    eval_eq("date('2004-01-01','+1 month')", text("2004-02-01"));
    eval_eq("date('2004-01-01','+1 year')", text("2005-01-01"));
}

#[test]
fn modifier_hours_and_minutes_combine_left_to_right() {
    eval_eq(
        "datetime('2004-01-01 12:00:00','+1 hour','+30 minutes')",
        text("2004-01-01 13:30:00"),
    );
}

#[test]
fn modifier_minutes_and_seconds_carry_into_hours() {
    eval_eq("datetime('2004-01-01 00:00:00','+90 minutes')", text("2004-01-01 01:30:00"));
    eval_eq("datetime('2004-01-01 00:00:00','+3600 seconds')", text("2004-01-01 01:00:00"));
}

#[test]
fn modifier_trailing_s_is_optional() {
    // §3: "The 's' character at the end of the modifier names in 1 through 6 is
    // optional." Singular and plural forms are equivalent.
    eval_eq("date('2004-01-01','+1 days')", text("2004-01-02"));
    eval_eq("date('2004-01-01','+2 month')", text("2004-03-01"));
    eval_eq("date('2004-01-01','+1 years')", text("2005-01-01"));
}

#[test]
fn modifier_amount_may_be_fractional() {
    // §3: "The NNN value can be any floating point number." +1.5 days = 1 day
    // and 12 hours.
    eval_eq("datetime('2004-01-01 00:00:00','+1.5 days')", text("2004-01-02 12:00:00"));
}

#[test]
fn modifier_signed_hh_mm_time_shift() {
    // §3 format 7: "±HH:MM" time-shift modifier (the leading sign is optional
    // for formats 7–9). +01:30 adds one hour and thirty minutes.
    eval_eq("datetime('2004-01-01 00:00:00','+01:30')", text("2004-01-01 01:30:00"));
}

#[test]
fn modifier_start_of_month() {
    // §3 modifiers 16–18 shift the date backward to the start of the unit.
    eval_eq("date('2004-01-15','start of month')", text("2004-01-01"));
}

#[test]
fn modifier_start_of_year() {
    eval_eq("date('2004-06-15','start of year')", text("2004-01-01"));
}

#[test]
fn modifier_start_of_day_zeroes_the_time() {
    eval_eq("datetime('2004-06-15 12:34:56','start of day')", text("2004-06-15 00:00:00"));
}

#[test]
fn modifier_last_day_of_month_idiom() {
    // The documented idiom (§4): start of month, +1 month, -1 day. For a
    // leap-year February this lands on the 29th.
    eval_eq(
        "date('2004-02-15','start of month','+1 month','-1 day')",
        text("2004-02-29"),
    );
    // A 31-day month yields the 31st.
    eval_eq(
        "date('2004-01-15','start of month','+1 month','-1 day')",
        text("2004-01-31"),
    );
}

// ---------------------------------------------------------------------------
// weekday N (§3 modifier 19): "advances the date forward, if necessary, to the
// next date where the weekday number is N. Sunday is 0 ... If the date is
// already on the desired weekday, [it] leaves the date unchanged."
// Reference weekdays: 2004-01-01 is a Thursday(4); 2004-01-04 is a Sunday(0).
// ---------------------------------------------------------------------------

#[test]
fn modifier_weekday_advances_to_next_matching_day() {
    // Thursday(4) advancing to the next Sunday(0) ⇒ 2004-01-04.
    eval_eq("date('2004-01-01','weekday 0')", text("2004-01-04"));
}

#[test]
fn modifier_weekday_on_matching_day_is_unchanged() {
    // 2004-01-01 is already Thursday(4), so "weekday 4" is a no-op.
    eval_eq("date('2004-01-01','weekday 4')", text("2004-01-01"));
    // 2004-01-04 is Sunday(0), so "weekday 0" is a no-op.
    eval_eq("date('2004-01-04','weekday 0')", text("2004-01-04"));
}

#[test]
fn modifier_weekday_advances_one_day() {
    // Sunday(0) advancing to the next Monday(1) ⇒ 2004-01-05.
    eval_eq("date('2004-01-04','weekday 1')", text("2004-01-05"));
}

// ---------------------------------------------------------------------------
// Month/year overflow and the ceiling/floor modifiers (§3, "dtambg"). The
// DEFAULT is "ceiling" (choose the later date); "floor" resolves to the last
// day of the previous month. Both worked examples come straight from the spec.
// ---------------------------------------------------------------------------

#[test]
fn modifier_month_overflow_default_is_ceiling() {
    // 2004-01-31 + 1 month ⇒ nominal 2004-02-31; February 2004 has 29 days, so
    // ceiling (the default) rolls the extra two days to 2004-03-02.
    eval_eq("date('2004-01-31','+1 month')", text("2004-03-02"));
}

#[test]
fn modifier_plus_one_year_onto_leap_day_ceiling_default() {
    // Spec example: "what is the date one year after 2024-02-29?" Ceiling
    // (default) ⇒ 2025-03-01.
    eval_eq("date('2024-02-29','+1 year')", text("2025-03-01"));
}

#[test]
fn modifier_plus_one_year_onto_leap_day_floor() {
    // Same spec example with "floor" ⇒ last day of the previous month,
    // 2025-02-28.
    eval_eq("date('2024-02-29','+1 year','floor')", text("2025-02-28"));
}

#[test]
fn modifier_plus_one_year_onto_leap_day_explicit_ceiling() {
    eval_eq("date('2024-02-29','+1 year','ceiling')", text("2025-03-01"));
}

#[test]
fn modifier_plus_two_months_onto_dec_31_ceiling_default() {
    // Spec example: "two months after 2023-12-31?" Ceiling (default) ⇒
    // 2024-03-02.
    eval_eq("date('2023-12-31','+2 months')", text("2024-03-02"));
}

#[test]
fn modifier_plus_two_months_onto_dec_31_floor() {
    // Same spec example with "floor" ⇒ 2024-02-29.
    eval_eq("date('2023-12-31','+2 months','floor')", text("2024-02-29"));
}

// ---------------------------------------------------------------------------
// 'unixepoch' modifier (§3 modifier 20): when it immediately follows a numeric
// (DDDDDDDDDD) time-value, that number is read as a Unix timestamp (seconds
// since 1970-01-01) rather than a Julian day number.
// ---------------------------------------------------------------------------

#[test]
fn unixepoch_modifier_reads_number_as_unix_seconds() {
    eval_eq("datetime(0,'unixepoch')", text("1970-01-01 00:00:00"));
    eval_eq("datetime(1072915200,'unixepoch')", text("2004-01-01 00:00:00"));
    eval_eq("date(1072915200,'unixepoch')", text("2004-01-01"));
}

// ---------------------------------------------------------------------------
// strftime() format codes (§1 substitution table). date(), time() and
// datetime() are exactly strftime('%F'), strftime('%T'), strftime('%F %T').
// ---------------------------------------------------------------------------

#[test]
fn strftime_compound_date_and_time() {
    eval_eq("strftime('%Y-%m-%d','2004-01-01 12:34:56')", text("2004-01-01"));
    eval_eq("strftime('%H:%M:%S','2004-01-01 12:34:56')", text("12:34:56"));
}

#[test]
fn strftime_year_month_day_are_zero_padded() {
    // %Y: year; %m: month 01-12; %d: day of month 01-31.
    eval_eq("strftime('%Y','2004-07-15')", text("2004"));
    eval_eq("strftime('%m','2004-07-15')", text("07"));
    eval_eq("strftime('%d','2004-07-05')", text("05"));
    eval_eq("strftime('%d','2004-07-15')", text("15"));
}

#[test]
fn strftime_hour_minute_second_are_zero_padded() {
    // %H: hour 00-24; %M: minute 00-59; %S: seconds 00-59.
    eval_eq("strftime('%H','2004-01-01 23:34:56')", text("23"));
    eval_eq("strftime('%H','2004-01-01 05:34:56')", text("05"));
    eval_eq("strftime('%M','2004-01-01 12:04:56')", text("04"));
    eval_eq("strftime('%S','2004-01-01 12:34:06')", text("06"));
}

#[test]
fn strftime_day_of_year_j_is_three_digit() {
    // %j: day of year 001-366.
    eval_eq("strftime('%j','2004-01-01')", text("001"));
    // 2004 is a leap year: 31 (Jan) + 29 = day 60 on 2004-02-29.
    eval_eq("strftime('%j','2004-02-29')", text("060"));
    // Last day of a leap year is day 366.
    eval_eq("strftime('%j','2004-12-31')", text("366"));
    // Last day of a non-leap year is day 365.
    eval_eq("strftime('%j','2003-12-31')", text("365"));
}

#[test]
fn strftime_weekday_w_sunday_is_zero() {
    // %w: day of week 0-6 with Sunday==0.
    eval_eq("strftime('%w','2004-01-04')", text("0")); // Sunday
    eval_eq("strftime('%w','2004-01-01')", text("4")); // Thursday
    eval_eq("strftime('%w','2004-01-05')", text("1")); // Monday
}

#[test]
fn strftime_weekday_u_monday_is_one() {
    // %u: day of week 1-7 with Monday==1 (so Sunday==7).
    eval_eq("strftime('%u','2004-01-05')", text("1")); // Monday
    eval_eq("strftime('%u','2004-01-04')", text("7")); // Sunday
}

#[test]
fn strftime_seconds_since_epoch_s() {
    // %s: seconds since 1970-01-01. §1 groups %s with %J as "the text
    // representation of the corresponding number", but unlike %J — a float whose
    // trailing-zero rendering the spec leaves unspecified — %s without `subsec`
    // is an INTEGER, and an integer's decimal text is canonical. So pinning the
    // exact text here is spec-faithful (it also checks the rendering, not just
    // the value), whereas the same exact-text pin on %J would over-specify.
    eval_eq("strftime('%s','1970-01-01 00:00:00')", text("0"));
    eval_eq("strftime('%s','2004-01-01 00:00:00')", text("1072915200"));
}

#[test]
fn strftime_iso_shorthands_f_t_r() {
    // %F: ISO 8601 date; %T: ISO 8601 time HH:MM:SS; %R: ISO 8601 time HH:MM.
    eval_eq("strftime('%F','2004-07-15 09:08:07')", text("2004-07-15"));
    eval_eq("strftime('%T','2004-07-15 09:08:07')", text("09:08:07"));
    eval_eq("strftime('%R','2004-07-15 09:08:07')", text("09:08"));
}

#[test]
fn strftime_percent_literal() {
    // %%: a literal percent sign.
    eval_eq("strftime('%%','2004-01-01')", text("%"));
}

#[test]
fn strftime_literal_text_passes_through() {
    // Characters that are not part of a substitution are copied verbatim.
    eval_eq("strftime('year=%Y','2004-01-01')", text("year=2004"));
    eval_eq("strftime('%Y/%m/%d','2004-07-05')", text("2004/07/05"));
}

#[test]
fn strftime_unsupported_substitution_is_null() {
    // §1: "If an undefined or unsupported substitution is seen, the result is
    // NULL." %Q is not a defined code.
    eval_eq("strftime('%Q','2004-01-01')", null());
}

// ---------------------------------------------------------------------------
// julianday() — REAL number of days since noon 4714-11-24 BC (§1). Compared
// with a small tolerance because it is a floating-point value.
// ---------------------------------------------------------------------------

#[test]
fn julianday_known_epochs() {
    // J2000: 2000-01-01 12:00:00 UTC is exactly JD 2451545.0.
    assert_scalar_approx(&mut mem(), "SELECT julianday('2000-01-01 12:00:00')", 2451545.0, 1e-6);
    // Unix epoch: 1970-01-01 00:00:00 UTC is JD 2440587.5.
    assert_scalar_approx(&mut mem(), "SELECT julianday('1970-01-01 00:00:00')", 2440587.5, 1e-6);
}

#[test]
fn julianday_midnight_is_half_day_before_noon() {
    // Midnight is 0.5 day before noon of the same date.
    assert_scalar_approx(&mut mem(), "SELECT julianday('2000-01-01 00:00:00')", 2451544.5, 1e-6);
    assert_scalar_approx(&mut mem(), "SELECT julianday('1970-01-01 12:00:00')", 2440588.0, 1e-6);
}

// ---------------------------------------------------------------------------
// unixepoch() — INTEGER seconds since 1970-01-01 00:00:00 UTC (§1). Every day
// is exactly 86400 seconds (§5, no leap seconds).
// ---------------------------------------------------------------------------

#[test]
fn unixepoch_known_instants() {
    eval_eq("unixepoch('1970-01-01 00:00:00')", int(0));
    eval_eq("unixepoch('1970-01-02 00:00:00')", int(86400));
    eval_eq("unixepoch('2000-01-01 00:00:00')", int(946684800));
    eval_eq("unixepoch('2004-01-01 00:00:00')", int(1072915200));
}

#[test]
fn unixepoch_roundtrips_via_unixepoch_modifier() {
    // Feeding a Unix timestamp back through the 'unixepoch' modifier returns it.
    eval_eq("unixepoch(1072915200,'unixepoch')", int(1072915200));
}

// ---------------------------------------------------------------------------
// NULL / invalid handling. A NULL time-value, or an uninterpretable string,
// yields NULL rather than an error.
// ---------------------------------------------------------------------------

#[test]
fn invalid_text_input_returns_null() {
    eval_eq("date('not a date')", null());
    eval_eq("time('not a time')", null());
    eval_eq("datetime('garbage')", null());
    eval_eq("julianday('nonsense')", null());
    eval_eq("unixepoch('nonsense')", null());
}

#[test]
fn null_time_value_propagates_to_null() {
    eval_eq("date(NULL)", null());
    eval_eq("time(NULL)", null());
    eval_eq("datetime(NULL)", null());
    eval_eq("julianday(NULL)", null());
    eval_eq("unixepoch(NULL)", null());
    eval_eq("strftime('%Y',NULL)", null());
}

// ---------------------------------------------------------------------------
// Intricate / higher-risk cases (§1 %f, §3 subsec). These are the trickiest
// corners of the spec (sub-second formatting), so they are isolated: assert the
// documented behavior and let a discrepancy surface as a single failing case.
// ---------------------------------------------------------------------------

#[test]
fn strftime_fractional_seconds_f() {
    // §1: "%f fractional seconds: SS.SSS" — seconds with a 3-digit fraction.
    eval_eq("strftime('%f','2004-01-01 12:34:56.789')", text("56.789"));
    // A shorter fraction is zero-extended to milliseconds.
    eval_eq("strftime('%f','2004-01-01 12:34:56.5')", text("56.500"));
    // No sub-second component ⇒ .000.
    eval_eq("strftime('%f','2004-01-01 12:34:56')", text("56.000"));
}

#[test]
fn subsec_modifier_adds_millisecond_resolution_to_text() {
    // §3 subsec: datetime()/time() gain a fractional-seconds field. The spec
    // hedges the width ("might increase to a higher resolution in future
    // releases") but documents the CURRENT implementation as milliseconds, so we
    // pin 3 digits deliberately — a future width change must be recorded here,
    // not silently accommodated by loosening the assertion.
    eval_eq("time('2004-01-01 12:34:56.789','subsec')", text("12:34:56.789"));
    eval_eq(
        "datetime('2004-01-01 12:34:56.5','subsec')",
        text("2004-01-01 12:34:56.500"),
    );
}

#[test]
fn subsec_modifier_makes_unixepoch_return_a_real() {
    // §3 subsec: "When 'subsec' is used with unixepoch(), the result is a
    // floating point value" — the fractional seconds since 1970.
    assert_scalar_approx(
        &mut mem(),
        "SELECT unixepoch('2004-01-01 00:00:00.5','subsec')",
        1072915200.5,
        1e-3,
    );
}

#[test]
fn strftime_julian_day_code_j_casts_to_the_julian_day() {
    // §1: "%J Julian day number (fractional)". The spec pins the VALUE but not
    // the exact text form — a whole-number Julian day may render without a
    // trailing ".0" (e.g. "2451545") — so testing a literal string here would
    // assert something the spec never states. The documented, format-agnostic
    // contract is the equivalence from §1: julianday(...) ≡
    // CAST(strftime('%J', ...) AS REAL). That is what we check.
    assert_scalar_approx(
        &mut mem(),
        "SELECT CAST(strftime('%J','2000-01-01 12:00:00') AS REAL)",
        2451545.0,
        1e-6,
    );
    // A fractional Julian day: 18:00 is 0.25 day after the JD's noon origin.
    assert_scalar_approx(
        &mut mem(),
        "SELECT CAST(strftime('%J','2000-01-01 18:00:00') AS REAL)",
        2451545.25,
        1e-6,
    );
    // A third INDEPENDENT absolute anchor (not an engine-vs-engine consistency
    // check): the Unix epoch is exactly Julian day 2440587.5.
    assert_scalar_approx(
        &mut mem(),
        "SELECT CAST(strftime('%J','1970-01-01 00:00:00') AS REAL)",
        2440587.5,
        1e-6,
    );
}

// ---------------------------------------------------------------------------
// 12-hour clock codes (§1): %I hour for the 12-hour clock (01-12, zero-padded);
// %p "AM"/"PM"; %P "am"/"pm". By the universal 12-hour convention the code's
// documented range (01-12) forces midnight to 12 (AM) and noon to 12 (PM).
// ---------------------------------------------------------------------------

#[test]
fn strftime_twelve_hour_hour_i() {
    eval_eq("strftime('%I','2004-01-01 00:00:00')", text("12")); // midnight ⇒ 12
    eval_eq("strftime('%I','2004-01-01 09:00:00')", text("09"));
    eval_eq("strftime('%I','2004-01-01 12:00:00')", text("12")); // noon ⇒ 12
    eval_eq("strftime('%I','2004-01-01 13:00:00')", text("01"));
}

#[test]
fn strftime_meridiem_upper_p() {
    eval_eq("strftime('%p','2004-01-01 00:00:00')", text("AM"));
    eval_eq("strftime('%p','2004-01-01 09:00:00')", text("AM"));
    eval_eq("strftime('%p','2004-01-01 12:00:00')", text("PM"));
    eval_eq("strftime('%p','2004-01-01 13:00:00')", text("PM"));
}

#[test]
fn strftime_meridiem_lower_p() {
    eval_eq("strftime('%P','2004-01-01 09:00:00')", text("am"));
    eval_eq("strftime('%P','2004-01-01 13:00:00')", text("pm"));
}

// ---------------------------------------------------------------------------
// 'auto' modifier (§3 automod): for a numeric time-value it selects Julian day
// vs Unix timestamp by magnitude — a Julian day number when the value is in
// [0.0, 5373484.499999], otherwise a Unix timestamp (within the wider numeric
// range). Both inputs below resolve to dates inside the functions' supported
// 0000–9999 window (§5), so the results are well-defined.
// ---------------------------------------------------------------------------

#[test]
fn auto_modifier_reads_small_number_as_julian_day() {
    // 2451545.0 is a valid Julian day number ⇒ interpreted as JD.
    eval_eq("datetime(2451545.0,'auto')", text("2000-01-01 12:00:00"));
}

#[test]
fn auto_modifier_reads_large_number_as_unix_timestamp() {
    // 1072915200 exceeds the maximum Julian day (5373484.5) ⇒ interpreted as a
    // Unix timestamp.
    eval_eq("datetime(1072915200,'auto')", text("2004-01-01 00:00:00"));
}

// ---------------------------------------------------------------------------
// 'julianday' modifier (§3 jdmod): forces a numeric time-value to be read as a
// Julian day number (the default meaning, so effectively a no-op). "Any other
// use of the 'julianday' modifier is an error and causes the function to return
// NULL" — in particular, applying it to a non-numeric (ISO text) time-value.
// ---------------------------------------------------------------------------

#[test]
fn julianday_modifier_is_a_noop_on_numeric_input() {
    eval_eq("datetime(2451545.0,'julianday')", text("2000-01-01 12:00:00"));
}

#[test]
fn julianday_modifier_on_non_numeric_input_is_null() {
    // The time-value is ISO text, not the DDDDDDDDDD form, so the 'julianday'
    // modifier is misused ⇒ NULL.
    eval_eq("date('2004-01-01','julianday')", null());
}

// ---------------------------------------------------------------------------
// 'subsec' with strftime %s (§3 subsec + §1 %s): subsec makes %s fractional —
// "a floating point value which is the number of seconds and fractional
// seconds since 1970". As with %J, the float's exact text form is unspecified,
// so the VALUE is checked via CAST-to-REAL rather than a literal string.
// ---------------------------------------------------------------------------

#[test]
fn strftime_seconds_with_subsec_is_fractional() {
    assert_scalar_approx(
        &mut mem(),
        "SELECT CAST(strftime('%s','2004-01-01 00:00:00.5','subsec') AS REAL)",
        1072915200.5,
        1e-3,
    );
}
