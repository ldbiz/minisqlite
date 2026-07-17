//! Date/time built-in functions (`date`, `time`, `datetime`, `julianday`,
//! `unixepoch`, `strftime`, `timediff`) per `spec/sqlite-doc/lang_datefunc.html`.
//!
//! The wall clock is read through [`FnContext::now_unix_millis`]; the `CURRENT_*`
//! keywords are lowered elsewhere and are deliberately not implemented here. Each
//! function is a zero-sized `ScalarFunction` that runs the shared
//! parse-time-value-then-apply-modifiers pipeline ([`compute`]) and renders the
//! result ([`render`]/[`strftime`]). A malformed time-value or an invalid modifier
//! yields SQL NULL, never an error — matching SQLite.
//!
//! Every function here overrides `ScalarFunction::deterministic()` to `false`: with a
//! `'now'` (or omitted) time-value they read the wall clock, so the result is not a pure
//! function of the arguments — exactly why SQLite does not mark the date/time family
//! `SQLITE_DETERMINISTIC`. This is conservative for a concrete time-value (`date(col)` is
//! in fact deterministic), but reporting the family non-deterministic only forgoes
//! memoizing a correlated subquery that uses one; the alternative — memoizing a
//! `datetime('now')` — risks a stale, wrong result.
//!
//! This module is the family's only wiring point (`register`) and owns the thin
//! function structs; the real work lives in the sibling submodules.

mod compute;
mod julian;
mod modifier;
mod parse;
mod render;
mod strftime;
mod value;

use std::sync::Arc;

use minisqlite_expr::{FnContext, ScalarFunction};
use minisqlite_types::{Result, Value};

use crate::registry::{Arity, FunctionRegistry};

/// Register the date/time function family. All of `date`/`time`/`datetime`/
/// `julianday`/`unixepoch` accept an optional time-value plus any number of modifiers
/// (`Arity::Any`, which includes the zero-argument 'now' form); `strftime` needs at
/// least the format string; `timediff` takes exactly two time-values.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_scalar("date", Arity::Any, Arc::new(DateFn));
    reg.add_scalar("time", Arity::Any, Arc::new(TimeFn));
    reg.add_scalar("datetime", Arity::Any, Arc::new(DateTimeFn));
    reg.add_scalar("julianday", Arity::Any, Arc::new(JulianDayFn));
    reg.add_scalar("unixepoch", Arity::Any, Arc::new(UnixEpochFn));
    reg.add_scalar("strftime", Arity::AtLeast(1), Arc::new(StrftimeFn));
    reg.add_scalar("timediff", Arity::Exact(2), Arc::new(TimeDiffFn));
}

/// `date(time-value?, modifier*)` -> `YYYY-MM-DD`.
#[derive(Debug)]
struct DateFn;
impl ScalarFunction for DateFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(compute::compute(args, ctx).map_or(Value::Null, |c| render::render_date(&c)))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

/// `time(time-value?, modifier*)` -> `HH:MM:SS[.SSS]`.
#[derive(Debug)]
struct TimeFn;
impl ScalarFunction for TimeFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(compute::compute(args, ctx).map_or(Value::Null, |c| render::render_time(&c)))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

/// `datetime(time-value?, modifier*)` -> `YYYY-MM-DD HH:MM:SS[.SSS]`.
#[derive(Debug)]
struct DateTimeFn;
impl ScalarFunction for DateTimeFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(compute::compute(args, ctx).map_or(Value::Null, |c| render::render_datetime(&c)))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

/// `julianday(time-value?, modifier*)` -> REAL Julian day.
#[derive(Debug)]
struct JulianDayFn;
impl ScalarFunction for JulianDayFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(compute::compute(args, ctx).map_or(Value::Null, |c| render::render_julianday(&c)))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

/// `unixepoch(time-value?, modifier*)` -> INTEGER (or REAL with `subsec`) Unix seconds.
#[derive(Debug)]
struct UnixEpochFn;
impl ScalarFunction for UnixEpochFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(compute::compute(args, ctx).map_or(Value::Null, |c| render::render_unixepoch(&c)))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

/// `strftime(format, time-value?, modifier*)`. The format is the first argument; the
/// rest form the ordinary time-value + modifier list. A NULL format, an unparseable
/// time-value, or an unsupported format code all yield NULL.
#[derive(Debug)]
struct StrftimeFn;
impl ScalarFunction for StrftimeFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        let fmt = match compute::text_of(&args[0]) {
            Some(f) => f,
            None => return Ok(Value::Null),
        };
        let computed = match compute::compute(&args[1..], ctx) {
            Some(c) => c,
            None => return Ok(Value::Null),
        };
        Ok(strftime::strftime(&fmt, &computed).map_or(Value::Null, Value::Text))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

/// `timediff(A, B)` -> the calendar difference to add to B to reach A. Both arguments
/// are time-values (no modifiers); either being NULL/unparseable yields NULL.
#[derive(Debug)]
struct TimeDiffFn;
impl ScalarFunction for TimeDiffFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        let a = match compute::resolve_single(&args[0], ctx) {
            Some(i) => i,
            None => return Ok(Value::Null),
        };
        let b = match compute::resolve_single(&args[1], ctx) {
            Some(i) => i,
            None => return Ok(Value::Null),
        };
        Ok(Value::Text(render::timediff(a, b)))
    }
    fn deterministic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic [`FnContext`] with a fixed clock, mirroring `scalar/misc.rs`.
    /// The clock is 1234567890000 ms == 2009-02-13 23:31:30 UTC so 'now' assertions
    /// are stable and cross-checkable.
    struct TestCtx;
    impl FnContext for TestCtx {
        fn now_unix_millis(&self) -> i64 {
            1_234_567_890_000
        }
        fn random_i64(&mut self) -> i64 {
            0
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn last_insert_rowid(&self) -> i64 {
            0
        }
        fn changes(&self) -> i64 {
            0
        }
        fn total_changes(&self) -> i64 {
            0
        }
    }

    fn call(f: &dyn ScalarFunction, args: &[Value]) -> Value {
        f.call(args, &mut TestCtx).expect("date/time call should not error")
    }

    fn t(s: &str) -> Value {
        Value::Text(s.into())
    }

    fn text(v: &Value) -> &str {
        match v {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn date_time_datetime_basic() {
        assert_eq!(text(&call(&DateFn, &[t("2009-02-13")])), "2009-02-13");
        assert_eq!(text(&call(&TimeFn, &[t("2009-02-13 23:31:30")])), "23:31:30");
        assert_eq!(text(&call(&DateTimeFn, &[t("2009-02-13 23:31:30")])), "2009-02-13 23:31:30");
    }

    #[test]
    fn datetime_from_unixepoch_modifier() {
        // Headline example.
        assert_eq!(
            text(&call(&DateTimeFn, &[Value::Integer(1_234_567_890), t("unixepoch")])),
            "2009-02-13 23:31:30"
        );
        // A numeric text time-value works the same way.
        assert_eq!(
            text(&call(&DateTimeFn, &[t("1234567890"), t("unixepoch")])),
            "2009-02-13 23:31:30"
        );
    }

    #[test]
    fn julianday_and_unixepoch_values() {
        match call(&JulianDayFn, &[t("2000-01-01")]) {
            Value::Real(r) => assert_eq!(r, 2451544.5),
            other => panic!("expected Real, got {other:?}"),
        }
        match call(&UnixEpochFn, &[t("2009-02-13 23:31:30")]) {
            Value::Integer(s) => assert_eq!(s, 1_234_567_890),
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn strftime_and_modifiers() {
        assert_eq!(text(&call(&StrftimeFn, &[t("%Y/%m/%d"), t("2009-02-13")])), "2009/02/13");
        assert_eq!(text(&call(&DateFn, &[t("2009-02-13"), t("+1 day")])), "2009-02-14");
        // Month-overflow through the public function surface.
        assert_eq!(text(&call(&DateFn, &[t("2009-01-31"), t("+1 month")])), "2009-03-03");
        assert_eq!(text(&call(&DateFn, &[t("2009-02-13"), t("start of month")])), "2009-02-01");
    }

    #[test]
    fn now_resolves_through_fixed_clock() {
        // 'now' == 2009-02-13 (from the fixed TestCtx clock).
        assert_eq!(text(&call(&DateFn, &[t("now")])), "2009-02-13");
        // Omitted time-value also means 'now'.
        assert_eq!(text(&call(&DateFn, &[])), "2009-02-13");
        assert_eq!(text(&call(&DateTimeFn, &[])), "2009-02-13 23:31:30");
        // strftime with only a format also uses 'now'.
        assert_eq!(text(&call(&StrftimeFn, &[t("%Y")])), "2009");
    }

    #[test]
    fn subsec_shortcut_and_resolution() {
        // unixepoch('subsec') -> fractional seconds of 'now' (a REAL).
        match call(&UnixEpochFn, &[t("subsec")]) {
            Value::Real(r) => assert!((r - 1_234_567_890.0).abs() < 1e-6),
            other => panic!("expected Real, got {other:?}"),
        }
        assert_eq!(text(&call(&TimeFn, &[t("2009-02-13 01:02:03.400"), t("subsec")])), "01:02:03.400");
    }

    #[test]
    fn localtime_is_utc_noop_limitation() {
        // LIMITATION: no OS tz database -> localtime/utc are no-ops (local == UTC).
        assert_eq!(
            text(&call(&DateTimeFn, &[t("2009-02-13 12:00:00"), t("localtime")])),
            "2009-02-13 12:00:00"
        );
        assert_eq!(
            text(&call(&DateTimeFn, &[t("2009-02-13 12:00:00"), t("utc")])),
            "2009-02-13 12:00:00"
        );
    }

    #[test]
    fn timediff_function() {
        assert_eq!(text(&call(&TimeDiffFn, &[t("2023-02-15"), t("2023-03-15")])), "-0000-01-00 00:00:00.000");
    }

    /// The exact examples worked in `lang_datefunc.html`, pinned end-to-end through the
    /// public function surface (the strongest anchor to documented behavior).
    #[test]
    fn documented_examples() {
        // "Compute the date and time given a unix timestamp 1092941466."
        assert_eq!(
            text(&call(&DateTimeFn, &[Value::Integer(1_092_941_466), t("unixepoch")])),
            "2004-08-19 18:51:06"
        );
        // "Compute the last day of the current month" (now = 2009-02).
        assert_eq!(
            text(&call(
                &DateFn,
                &[t("now"), t("start of month"), t("+1 month"), t("-1 day")]
            )),
            "2009-02-28"
        );
        // "Compute the date of the first Tuesday in October for the current year."
        assert_eq!(
            text(&call(&DateFn, &[t("now"), t("start of year"), t("+9 months"), t("weekday 2")])),
            "2009-10-06"
        );
        // "Compute how old Abraham Lincoln would be if he were still alive today":
        // timediff('now','1809-02-12') against the fixed 2009-02-13 23:31:30 clock.
        assert_eq!(
            text(&call(&TimeDiffFn, &[t("now"), t("1809-02-12")])),
            "+0200-00-01 23:31:30.000"
        );
    }

    #[test]
    fn julianday_modifier_requires_numeric_time_value() {
        // 'julianday' on a numeric time-value is a near no-op (renders the date)...
        assert_eq!(
            text(&call(&DateFn, &[Value::Real(2451545.0), t("julianday")])),
            "2000-01-01"
        );
        // ...but on an ISO text time-value it is an error -> NULL (spec: jdmod).
        assert!(matches!(call(&DateFn, &[t("2000-01-01"), t("julianday")]), Value::Null));
        // 'auto' by magnitude: a unix-range number becomes a unix timestamp.
        assert_eq!(
            text(&call(&DateTimeFn, &[Value::Integer(1_092_941_466), t("auto")])),
            "2004-08-19 18:51:06"
        );
        // A bare number defaults to a Julian day even for unixepoch() (spec: all
        // functions treat DDDDDDDDDD as a Julian day unless auto/unixepoch is added).
        match call(&JulianDayFn, &[Value::Real(2451545.0)]) {
            Value::Real(r) => assert_eq!(r, 2451545.0),
            other => panic!("expected Real, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_day_renders_raw_but_normalizes_under_a_modifier() {
        // An in-range-but-impossible day is stored verbatim and the FIELD renders show it
        // literally (SQLite: date ≡ strftime('%F'); the field codes read the stored day).
        assert_eq!(text(&call(&DateFn, &[t("2020-09-31")])), "2020-09-31");
        assert_eq!(text(&call(&DateTimeFn, &[t("2009-02-31")])), "2009-02-31 00:00:00");
        assert_eq!(text(&call(&StrftimeFn, &[t("%Y-%m-%d"), t("2020-09-31")])), "2020-09-31");
        assert_eq!(text(&call(&DateTimeFn, &[t("2009-02-31 12:34:56")])), "2009-02-31 12:34:56");
        // Mixed strftime: the field code %d shows the stored day 31, while the Julian-day
        // code %j uses the normalized instant (Feb 31 2009 -> Mar 3, day-of-year 62).
        assert_eq!(text(&call(&StrftimeFn, &[t("%d/%j"), t("2009-02-31")])), "31/062");
        assert_eq!(text(&call(&StrftimeFn, &[t("%d"), t("2009-02-31")])), "31");
        // Any modifier (even a zero shift) forces the Julian day and normalizes -> Mar 3.
        assert_eq!(text(&call(&DateFn, &[t("2009-02-31"), t("+0 days")])), "2009-03-03");
        // A Julian-day output normalizes regardless of modifiers.
        assert_eq!(
            call_int(&UnixEpochFn, &[t("2009-02-31")]),
            call_int(&UnixEpochFn, &[t("2009-03-03")])
        );
    }

    #[test]
    fn hour_24_renders_literally_without_rolling_the_date() {
        // `%H` is 00-24: a literal hour 24 is preserved and does NOT roll the date on a
        // field render — the same stored-field rule that keeps an out-of-range day
        // literal. So both fields stay literal together.
        assert_eq!(text(&call(&DateTimeFn, &[t("2009-02-28 24:00")])), "2009-02-28 24:00:00");
        assert_eq!(text(&call(&TimeFn, &[t("2009-02-28 24:00")])), "24:00:00");
        assert_eq!(text(&call(&DateTimeFn, &[t("2009-02-31 24:00")])), "2009-02-31 24:00:00");
    }

    fn call_int(f: &dyn ScalarFunction, args: &[Value]) -> i64 {
        match call(f, args) {
            Value::Integer(i) => i,
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn malformed_time_value_is_null() {
        // Pin: a garbage date is NULL, not an error.
        assert!(matches!(call(&DateFn, &[t("not a date")]), Value::Null));
        assert!(matches!(call(&DateFn, &[Value::Null]), Value::Null));
        // A lowercase 't' date/time separator is not accepted -> NULL (uppercase 'T'
        // and lowercase 'z' zone do parse). The same policy holds in a ±date modifier
        // offset, so the two separator sites agree.
        assert!(matches!(call(&DateFn, &[t("2009-02-13t00:00")]), Value::Null));
        assert_eq!(text(&call(&DateTimeFn, &[t("2009-02-13T00:00")])), "2009-02-13 00:00:00");
        assert!(matches!(
            call(&DateTimeFn, &[t("2000-01-01"), t("+0000-00-01t12:00")]),
            Value::Null
        ));
        assert_eq!(
            text(&call(&DateTimeFn, &[t("2000-01-01"), t("+0000-00-01T12:00")])),
            "2000-01-02 12:00:00"
        );
        // An invalid modifier is NULL.
        assert!(matches!(call(&DateFn, &[t("2009-02-13"), t("+1 fortnight")]), Value::Null));
        // A NULL modifier is NULL.
        assert!(matches!(call(&DateFn, &[t("2009-02-13"), Value::Null]), Value::Null));
        // strftime with a NULL format / unsupported code is NULL.
        assert!(matches!(call(&StrftimeFn, &[Value::Null, t("2009-02-13")]), Value::Null));
        assert!(matches!(call(&StrftimeFn, &[t("%Q"), t("2009-02-13")]), Value::Null));
    }

    #[test]
    fn datetime_family_is_nondeterministic() {
        // Every date/time function reads the wall clock for 'now', so the whole family
        // reports non-deterministic — matching SQLite's refusal to mark them
        // SQLITE_DETERMINISTIC. Used to keep a correlated subquery containing one out of
        // the memoization cache.
        assert!(!DateFn.deterministic());
        assert!(!TimeFn.deterministic());
        assert!(!DateTimeFn.deterministic());
        assert!(!JulianDayFn.deterministic());
        assert!(!UnixEpochFn.deterministic());
        assert!(!StrftimeFn.deterministic());
        assert!(!TimeDiffFn.deterministic());
    }

    #[test]
    fn registered_names_resolve() {
        let reg = FunctionRegistry::builtins();
        for (name, argc) in [
            ("date", 0usize),
            ("time", 1),
            ("datetime", 2),
            ("julianday", 1),
            ("unixepoch", 1),
            ("STRFTIME", 2),
            ("timediff", 2),
        ] {
            assert!(reg.resolve_scalar(name, argc).is_ok(), "{name}/{argc} should resolve");
        }
    }
}
