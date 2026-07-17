//! The shared pipeline the date/time functions run: turn the argument list into a
//! final [`Computed`] instant (plus the `subsec` flag), or `None` for a NULL result.
//!
//! All six of `date`/`time`/`datetime`/`julianday`/`unixepoch`/`strftime` differ only
//! in how they *render* a computed instant; the parse-time-value-then-apply-modifiers
//! work is identical and lives here. `timediff` does not take modifiers, so it uses
//! the lighter [`resolve_single`].

use minisqlite_types::{integer_to_text, real_to_text, Value};

use minisqlite_expr::FnContext;

use super::julian::Instant;
use super::modifier;
use super::parse::{self, TimeValue};
use super::value::DateTime;

/// A computed value plus whether a `subsec`/`subsecond` modifier was seen (which
/// raises the output resolution of `time`/`datetime`/`unixepoch`/`%s` to ms). The
/// [`DateTime`] may still be un-normalized: with no modifier, an out-of-range ISO day
/// and a literal hour 24 render verbatim, matching SQLite's deferred Julian-day compute.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Computed {
    pub(crate) dt: DateTime,
    pub(crate) subsec: bool,
}

/// Run the pipeline for the modifier-taking functions over `args` (for `strftime`
/// these are the arguments *after* the format string). An empty list, or a leading
/// `subsec`/`subsecond` in the time-value slot, both mean the time-value is 'now'.
pub(crate) fn compute(args: &[Value], ctx: &mut dyn FnContext) -> Option<Computed> {
    let (tv, mod_args): (TimeValue, &[Value]) = match args.first() {
        None => (TimeValue::Now, &[]),
        Some(first) => {
            // `subsec`/`subsecond` may stand in the first (time-value) slot, in which
            // case the time-value is 'now' and this argument is a modifier.
            if is_subsec_text(first) {
                (TimeValue::Now, args)
            } else {
                (parse::parse_time_value(first)?, &args[1..])
            }
        }
    };
    // Resolve 'now' here, where the clock is in scope; the applier is clock-free. 'now'
    // is a true instant, so it is already normalized.
    let tv = match tv {
        TimeValue::Now => {
            TimeValue::Resolved(DateTime::Normalized(Instant::from_now(ctx.now_unix_millis())))
        }
        other => other,
    };
    modifier::apply(tv, mod_args)
}

/// Resolve a single time-value with no modifiers, for `timediff`. A numeric value is
/// a Julian day (timediff cannot use `unixepoch`/`auto` since it takes no modifiers).
pub(crate) fn resolve_single(v: &Value, ctx: &mut dyn FnContext) -> Option<Instant> {
    match parse::parse_time_value(v)? {
        TimeValue::Now => Some(Instant::from_now(ctx.now_unix_millis())),
        // timediff computes a Julian-day difference, so both operands become instants
        // (normalizing any out-of-range day / literal hour-24 here).
        TimeValue::Resolved(dt) => Some(dt.to_instant()),
        TimeValue::RawNumber(r) => Some(Instant::from_julian_day(r)),
    }
}

/// The text form of a `Value` for modifier/format parsing. NULL yields `None` (NULL
/// result). Numbers render to their canonical text; a blob is decoded lossily.
pub(crate) fn text_of(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Text(s) => Some(s.clone()),
        Value::Integer(i) => Some(integer_to_text(*i)),
        Value::Real(r) => Some(real_to_text(*r)),
        Value::Blob(b) => Some(String::from_utf8_lossy(b).into_owned()),
    }
}

/// Whether `v` is the text `subsec`/`subsecond` (case-insensitive), which may occupy
/// the time-value slot as a shorthand for `'now', 'subsec'`.
fn is_subsec_text(v: &Value) -> bool {
    matches!(v, Value::Text(s) if {
        let t = s.trim();
        t.eq_ignore_ascii_case("subsec") || t.eq_ignore_ascii_case("subsecond")
    })
}
