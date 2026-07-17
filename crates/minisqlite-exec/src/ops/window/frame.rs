//! The window FRAME engine (windowfunctions.html §2.2).
//!
//! Given a [`Partition`], a current ordered position `c`, and a *resolved* frame
//! ([`ResolvedFrame`] — the plan's `ROWS`/`RANGE`/`GROUPS` frame with its `PRECEDING`/
//! `FOLLOWING` bound expressions already evaluated to non-negative numeric [`Value`]s by
//! the shell), [`frame_positions`] produces the [`Frame`]: the set of ordered positions
//! that make up the window frame AFTER applying `EXCLUDE`, in window order.
//!
//! # Why a resolved frame (functional core, imperative shell)
//!
//! Bound expressions are constant, so the shell ([`super`]) evaluates and validates them
//! ONCE and hands this engine plain values. Everything here is then a pure function of
//! `(partition, c, frame)` — no `Runtime`, no evaluator, no I/O — so the whole subtle
//! boundary/`EXCLUDE` calculus is exhaustively unit-testable against the spec's worked
//! examples (see the tests below, which transcribe the §2.2 tables verbatim).
//!
//! # The position model
//!
//! Each boundary resolves to a raw signed ordered position (which may fall outside
//! `[0, len)`); the frame is then the intersection with `[0, len-1]`, and it is EMPTY
//! when that intersection is empty (start past the end, end before the start, or a
//! boundary form that reports "no such row"). This one model yields SQLite's documented
//! clamping — a `PRECEDING` start that underflows clamps to the first row, a `FOLLOWING`
//! end that overflows clamps to the last row — AND the correct empty frames — a
//! `FOLLOWING`-only frame past the partition end, a `PRECEDING`-only frame before its
//! start — without special-casing either.

use minisqlite_expr::SortKey;
use minisqlite_types::Value;

use super::partition::Partition;

/// The frame TYPE: how boundaries are measured (windowfunctions.html §2.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameUnits {
    /// Count individual rows relative to the current row.
    Rows,
    /// Frame by the band of `ORDER BY` values around the current row's value.
    Range,
    /// Count peer groups relative to the current group.
    Groups,
}

/// The `EXCLUDE` clause (windowfunctions.html §2.2.3). "Peers" here are same-`ORDER BY`
/// value rows EVEN for the `ROWS` frame type, and the whole partition when there is no
/// `ORDER BY` — exactly [`Partition::peer_bounds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameExclude {
    /// The default: exclude nothing.
    NoOthers,
    /// Exclude the current row only (its peers remain).
    CurrentRow,
    /// Exclude the current row and all its peers.
    Group,
    /// Exclude the current row's peers but KEEP the current row.
    Ties,
}

/// A resolved frame boundary: `PRECEDING`/`FOLLOWING` carry the offset already evaluated
/// to a non-negative numeric [`Value`] (integer for `ROWS`/`GROUPS`, integer or real for
/// `RANGE`). Validation happens in the shell; this engine assumes a valid value.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedBound {
    UnboundedPreceding,
    Preceding(Value),
    CurrentRow,
    Following(Value),
    UnboundedFollowing,
}

/// A frame with its bound expressions already resolved. Built by the shell from the
/// plan's `WindowFrame`; consumed by [`frame_positions`].
#[derive(Debug, Clone)]
pub(crate) struct ResolvedFrame {
    pub(crate) units: FrameUnits,
    pub(crate) start: ResolvedBound,
    pub(crate) end: ResolvedBound,
    pub(crate) exclude: FrameExclude,
}

impl ResolvedFrame {
    /// Whether this is SQLite's DEFAULT frame — `RANGE BETWEEN UNBOUNDED PRECEDING AND
    /// CURRENT ROW EXCLUDE NO OTHERS`. The shell routes the default frame to the fast
    /// O(p) running-aggregate path; every other frame uses the general per-row path.
    pub(crate) fn is_default(&self) -> bool {
        self.units == FrameUnits::Range
            && matches!(self.start, ResolvedBound::UnboundedPreceding)
            && matches!(self.end, ResolvedBound::CurrentRow)
            && self.exclude == FrameExclude::NoOthers
    }
}

/// A computed window frame: the inclusive ordered-position range plus the information
/// needed to apply `EXCLUDE` lazily. Yields positions in WINDOW ORDER.
///
/// Navigation helpers ([`Frame::first`] / [`Frame::last`] / [`Frame::nth`]) are read by
/// [`super::navigation::value_fn`] for `first_value`/`last_value`/`nth_value`; the aggregate
/// kind folds over [`Frame::positions`].
pub(crate) struct Frame {
    /// Inclusive `[lo, hi]` range of ordered positions, or `None` for an empty frame.
    range: Option<(usize, usize)>,
    exclude: FrameExclude,
    /// Current row's ordered position and its peer group `[peer_lo, peer_hi)`.
    c: usize,
    peer_lo: usize,
    peer_hi: usize,
}

impl Frame {
    /// The frame's ordered positions in window order, after `EXCLUDE`.
    pub(crate) fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        // An empty frame becomes `1..=0`, which yields nothing (no allocation).
        let (lo, hi) = self.range.unwrap_or((1, 0));
        (lo..=hi).filter(move |&p| !self.is_excluded(p))
    }

    /// Whether ordered position `p` is removed by this frame's `EXCLUDE` clause.
    fn is_excluded(&self, p: usize) -> bool {
        match self.exclude {
            FrameExclude::NoOthers => false,
            FrameExclude::CurrentRow => p == self.c,
            FrameExclude::Group => self.peer_lo <= p && p < self.peer_hi,
            FrameExclude::Ties => self.peer_lo <= p && p < self.peer_hi && p != self.c,
        }
    }

    /// The frame's positions as up to three disjoint inclusive `[lo, hi]` segments in
    /// window order — the O(1) STRUCTURAL form of [`Frame::positions`]. It is the base
    /// `[lo, hi]` range with the `EXCLUDE` region removed: nothing (`NO OTHERS`), the single
    /// current row (`CURRENT ROW`), the current row's whole peer run (`GROUP`), or that run
    /// with the kept current row punched back in as a middle segment (`TIES`). The navigation
    /// helpers read this so `last`/`nth` are O(1) arithmetic rather than a walk of the whole
    /// frame (a `positions().last()`/`.nth()` walk made `last_value`/`nth_value` Θ(n²) over a
    /// partition). Equivalence with the tested `positions()`/`is_excluded` path is pinned by
    /// the `navigation_helpers_match_positions` unit test.
    fn segments(&self) -> [Option<(usize, usize)>; 3] {
        let (lo, hi) = match self.range {
            Some(r) => r,
            None => return [None; 3],
        };
        // An inclusive `[a, b]` segment, or `None` when it is empty (`a > b`).
        let seg = |a: usize, b: usize| (a <= b).then_some((a, b));
        match self.exclude {
            FrameExclude::NoOthers => [seg(lo, hi), None, None],
            // Remove the single current-row position, leaving its two flanks.
            FrameExclude::CurrentRow if (lo..=hi).contains(&self.c) => {
                let left = if self.c > lo { seg(lo, self.c - 1) } else { None };
                let right = if self.c < hi { seg(self.c + 1, hi) } else { None };
                [left, right, None]
            }
            FrameExclude::CurrentRow => [seg(lo, hi), None, None],
            FrameExclude::Group | FrameExclude::Ties => {
                // The current row's peer run `[peer_lo, peer_hi)` clamped to the frame.
                // `peer_hi >= 1` whenever `range` is `Some` (a non-empty partition always has
                // a real peer group), so `peer_hi - 1` cannot underflow.
                let run_lo = lo.max(self.peer_lo);
                let run_hi = hi.min(self.peer_hi - 1);
                if run_lo > run_hi {
                    // The peer run does not overlap the frame ⇒ nothing is excluded.
                    return [seg(lo, hi), None, None];
                }
                let left = if run_lo > lo { seg(lo, run_lo - 1) } else { None };
                let right = if run_hi < hi { seg(run_hi + 1, hi) } else { None };
                // TIES keeps the current row: punch it back in as a middle segment (it lies
                // inside the excluded run, between `left` and `right`). GROUP drops it too.
                let mid = match self.exclude {
                    FrameExclude::Ties if (lo..=hi).contains(&self.c) => seg(self.c, self.c),
                    _ => None,
                };
                [left, mid, right]
            }
        }
    }

    // `first`/`last`/`nth` are the O(1) navigation primitives the
    // `first_value`/`last_value`/`nth_value` kinds read via `super::navigation::value_fn`;
    // each reads `segments()` (at most three flanks) rather than walking `positions()`.
    // (`is_empty` below is exercised only by the unit tests — no non-test call site — so it
    // keeps `allow(dead_code)`; dropping that attribute if a real caller appears is how we
    // notice a helper going unused.)

    /// The first frame position in window order (post-`EXCLUDE`), or `None` if empty. O(1).
    pub(crate) fn first(&self) -> Option<usize> {
        self.segments().into_iter().flatten().next().map(|(lo, _)| lo)
    }

    /// The last frame position in window order (post-`EXCLUDE`), or `None` if empty. O(1) —
    /// reads at most three segments, NOT a walk of the whole frame.
    pub(crate) fn last(&self) -> Option<usize> {
        self.segments().into_iter().flatten().last().map(|(_, hi)| hi)
    }

    /// The `n`-th (0-BASED) frame position in window order (post-`EXCLUDE`), or `None`.
    /// SQL's `nth_value(e, N)` is 1-based, so a caller passes `N - 1` here. O(1) — indexes
    /// into the (≤3) segment lengths rather than walking `n` positions.
    pub(crate) fn nth(&self, n: usize) -> Option<usize> {
        let mut remaining = n;
        for (lo, hi) in self.segments().into_iter().flatten() {
            let count = hi - lo + 1;
            if remaining < count {
                return Some(lo + remaining);
            }
            remaining -= count;
        }
        None
    }

    /// Whether the frame contains no rows (after `EXCLUDE`).
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.first().is_none()
    }
}

/// Whether a bound is an `<expr> PRECEDING`/`FOLLOWING` offset (as opposed to
/// `UNBOUNDED …`/`CURRENT ROW`). Used to scope the RANGE single-ORDER-BY-term assertion.
fn has_offset_bound(b: &ResolvedBound) -> bool {
    matches!(b, ResolvedBound::Preceding(_) | ResolvedBound::Following(_))
}

/// Compute the frame at ordered position `c`. `order_by` is the window `ORDER BY` (only
/// its first term's direction is read, and only for the `RANGE` type).
///
/// The planner guarantees the start boundary does not appear "after" the end boundary in
/// the boundary ordering, but an EMPTY frame (start > end) is still handled here without
/// panicking — a bounded read that returns the empty set.
pub(crate) fn frame_positions(
    part: &Partition,
    c: usize,
    frame: &ResolvedFrame,
    order_by: &[SortKey],
) -> Frame {
    let (peer_lo, peer_hi) = part.peer_bounds(c);
    let len = part.len();
    if len == 0 {
        return Frame { range: None, exclude: frame.exclude, c, peer_lo, peer_hi };
    }

    let desc = order_by.first().map(|k| k.desc).unwrap_or(false);

    // A RANGE frame with an `<expr>` offset bound (`PRECEDING`/`FOLLOWING`) requires
    // EXACTLY one ORDER BY term (spec §2.2.2 — the band is measured against that single
    // value); the planner enforces this. Assert in debug so a planner regression surfaces
    // at its cause here, not as a subtly-wrong bound far downstream. (RANGE with only
    // UNBOUNDED/CURRENT ROW bounds uses peer semantics and needs no such restriction.)
    debug_assert!(
        frame.units != FrameUnits::Range
            || !(has_offset_bound(&frame.start) || has_offset_bound(&frame.end))
            || order_by.len() == 1,
        "RANGE offset frame requires exactly one ORDER BY term, got {}",
        order_by.len(),
    );

    let start = resolve_start(part, c, frame, desc);
    let end = resolve_end(part, c, frame, desc);

    let range = match (start, end) {
        (Some(s), Some(e)) => {
            // Intersect the raw signed boundaries with [0, len-1]; empty if disjoint.
            let lo = s.max(0);
            let hi = e.min(len as i64 - 1);
            if lo > hi {
                None
            } else {
                Some((lo as usize, hi as usize))
            }
        }
        _ => None,
    };

    Frame { range, exclude: frame.exclude, c, peer_lo, peer_hi }
}

/// The raw signed ordered position of the START boundary, or `None` when the boundary
/// reports "no such row" (an empty frame from this side).
fn resolve_start(part: &Partition, c: usize, frame: &ResolvedFrame, desc: bool) -> Option<i64> {
    match &frame.start {
        ResolvedBound::UnboundedPreceding => Some(0),
        // Degenerate (the planner forbids it as a start bound); treat as the last row.
        ResolvedBound::UnboundedFollowing => Some(part.len() as i64 - 1),
        ResolvedBound::CurrentRow => Some(match frame.units {
            FrameUnits::Rows => c as i64,
            FrameUnits::Range | FrameUnits::Groups => part.peer_bounds(c).0 as i64,
        }),
        ResolvedBound::Preceding(e) => start_offset(part, c, frame.units, e, false, desc),
        ResolvedBound::Following(e) => start_offset(part, c, frame.units, e, true, desc),
    }
}

/// The raw signed ordered position of the END boundary, or `None` for an empty frame.
fn resolve_end(part: &Partition, c: usize, frame: &ResolvedFrame, desc: bool) -> Option<i64> {
    match &frame.end {
        ResolvedBound::UnboundedFollowing => Some(part.len() as i64 - 1),
        // Degenerate (the planner forbids it as an end bound); treat as the first row.
        ResolvedBound::UnboundedPreceding => Some(0),
        ResolvedBound::CurrentRow => Some(match frame.units {
            FrameUnits::Rows => c as i64,
            FrameUnits::Range | FrameUnits::Groups => part.peer_bounds(c).1 as i64 - 1,
        }),
        ResolvedBound::Preceding(e) => end_offset(part, c, frame.units, e, false, desc),
        ResolvedBound::Following(e) => end_offset(part, c, frame.units, e, true, desc),
    }
}

/// A START `<expr> PRECEDING`/`FOLLOWING` boundary. `following` selects the sign of the
/// offset (`FOLLOWING` = +, `PRECEDING` = -).
fn start_offset(
    part: &Partition,
    c: usize,
    units: FrameUnits,
    e: &Value,
    following: bool,
    desc: bool,
) -> Option<i64> {
    match units {
        FrameUnits::Rows => {
            let off = signed_rows(e, following);
            // Saturating: a huge FOLLOWING (or PRECEDING) offset must clamp at the frame's
            // [0, len-1] intersection, never overflow (panics under overflow-checks).
            Some((c as i64).saturating_add(off))
        }
        FrameUnits::Groups => {
            let off = signed_rows(e, following);
            let g = (part.group_index(c) as i64).saturating_add(off);
            // Start past the last group ⇒ empty. Underflow ⇒ first group (clamp low).
            if g > part.num_groups() as i64 - 1 {
                return None;
            }
            let g = g.max(0) as usize;
            Some(part.group_bounds(g).0 as i64)
        }
        FrameUnits::Range => range_bound(part, c, e, following, desc, /* is_start */ true),
    }
}

/// An END `<expr> PRECEDING`/`FOLLOWING` boundary.
fn end_offset(
    part: &Partition,
    c: usize,
    units: FrameUnits,
    e: &Value,
    following: bool,
    desc: bool,
) -> Option<i64> {
    match units {
        FrameUnits::Rows => {
            let off = signed_rows(e, following);
            // Saturating: see `start_offset` — clamp a huge offset, never overflow.
            Some((c as i64).saturating_add(off))
        }
        FrameUnits::Groups => {
            let off = signed_rows(e, following);
            let g = (part.group_index(c) as i64).saturating_add(off);
            // End before the first group ⇒ empty. Overflow ⇒ last group (clamp high).
            if g < 0 {
                return None;
            }
            let g = g.min(part.num_groups() as i64 - 1) as usize;
            Some(part.group_bounds(g).1 as i64 - 1)
        }
        FrameUnits::Range => range_bound(part, c, e, following, desc, /* is_start */ false),
    }
}

/// A `RANGE` boundary: the extent of a band of `ORDER BY` values around the current
/// row's value (windowfunctions.html §2.2.2, `<expr> PRECEDING`, RANGE case).
///
/// With `Xc` the current row's single `ORDER BY` value and the signed offset `o`
/// (`FOLLOWING` positive, `PRECEDING` negative), a START boundary is the FIRST ordered
/// position whose value is in-band and an END boundary is the LAST such position:
/// * ASC: in-band-start `Xi >= Xc + o`, in-band-end `Xi <= Xc + o`.
/// * DESC (values descend in window order, mirrored): start `Xi <= Xc - o`, end
///   `Xi >= Xc - o`.
/// Non-numeric `Xi` are never in-band (they sort outside the numeric block, so the
/// contiguous numeric band is preserved). When `Xc` itself is non-numeric the bound is
/// by IS-equality — the current row's peer group — per the spec.
///
/// # O(log n) via binary search over the numeric run
///
/// The partition is sorted by this single `ORDER BY` value, and the storage-class order
/// (NULL < numeric[INTEGER/REAL by value] < TEXT < BLOB — `minisqlite_types::compare_values`)
/// makes the numeric values ONE contiguous run in window order, monotonic (non-decreasing
/// for ASC, non-increasing for DESC). Over that run `in_band` is therefore a true-SUFFIX
/// for a START bound (its first in-band position is the boundary) and a true-PREFIX for an
/// END bound (its last in-band position is the boundary) — each a partition point found by
/// BINARY SEARCH. That keeps a per-partition RANGE frame O(n log n) instead of the O(n²) a
/// per-row full scan costs. `c` is always inside the run here (a non-numeric `Xc` took the
/// peer path above), so it anchors the two searches that locate the run's `[num_lo, num_hi)`.
///
/// LIMITATION (vs real `sqlite3`, pre-existing and out of scope): the band arithmetic is
/// done in `f64`, so a `RANGE` offset against integer order values beyond 2^53 loses
/// precision; the common integer/real ranges are exact.
///
/// That precision limit does NOT weaken byte-identity with the linear reference
/// (`range_bound_linear`), which holds for EVERY input: `numeric_f64(v)` equals
/// `round_to_f64(exact_value(v))` — the identity on a REAL (already an `f64`) and
/// round-to-nearest on an integer — and round-to-nearest is globally monotonic. So the run,
/// exact-sorted by `compare_values`, stays monotonic in `f64` even for interleaved
/// INTEGER/REAL past 2^53; and a strict exact inequality that collapses to an `f64` tie gives
/// both tied rows the same `in_band`, so the scan and the search pick the same position.
/// `range_bound_binary_search_matches_linear_scan` pins this with values at/astride 2^53.
fn range_bound(
    part: &Partition,
    c: usize,
    e: &Value,
    following: bool,
    desc: bool,
    is_start: bool,
) -> Option<i64> {
    let xc = match numeric_at(part, c) {
        Some(x) => x,
        // Non-numeric (or absent) current value ⇒ peer semantics.
        None => {
            let (lo, hi) = part.peer_bounds(c);
            return Some(if is_start { lo as i64 } else { hi as i64 - 1 });
        }
    };
    let off = if following { numeric_f64(e).unwrap_or(0.0) } else { -numeric_f64(e).unwrap_or(0.0) };
    // The comparison threshold and direction, per ASC/DESC and start/end.
    let threshold = if desc { xc - off } else { xc + off };

    let in_band = |xi: f64| -> bool {
        match (desc, is_start) {
            (false, true) => xi >= threshold,  // ASC start
            (false, false) => xi <= threshold, // ASC end
            (true, true) => xi <= threshold,   // DESC start
            (true, false) => xi >= threshold,  // DESC end
        }
    };

    // Locate the contiguous numeric run `[num_lo, num_hi)` that contains `c`. "Is numeric"
    // is a true-suffix over `[0, c]` (so its first true is `num_lo`) and a true-prefix over
    // `[c, len)` (so its first false is `num_hi`); both hold because the numeric values are
    // one contiguous block and `c` itself is numeric ⇒ `num_lo <= c < num_hi` (never empty).
    let len = part.len();
    let num_lo = first_true(0, c + 1, |p| numeric_at(part, p).is_some());
    let num_hi = first_true(c, len, |p| numeric_at(part, p).is_none());

    // The search is load-bearing on numerics being EXACTLY the contiguous run
    // `[num_lo, num_hi)` — a property established UPSTREAM by `Partition::new`'s storage-class
    // sort, not here. Assert it in debug so a future break in that cross-module invariant (a
    // new numeric-ish storage class, a sort-comparator change) fails loudly at its cause
    // instead of silently producing a wrong frame. O(n), compiled out of release.
    debug_assert!(
        (0..len).all(|p| numeric_at(part, p).is_some() == (num_lo..num_hi).contains(&p)),
        "range_bound: numeric ORDER BY values must form the single contiguous run \
         [{num_lo}, {num_hi})",
    );

    // `in_band` over `[num_lo, num_hi)` is partitioned (SUFFIX for START, PREFIX for END).
    // A non-numeric position is never in-band — matching the linear reference and keeping
    // this total even if the run somehow held one (it never does for a sorted partition).
    let in_band_at = |p: usize| numeric_at(part, p).map(|xi| in_band(xi)).unwrap_or(false);
    if is_start {
        // First in-band position; `num_hi` (none in-band) ⇒ empty from this side.
        let p = first_true(num_lo, num_hi, &in_band_at);
        (p < num_hi).then_some(p as i64)
    } else {
        // Last in-band position = one before the first non-in-band; `num_lo` ⇒ none in-band.
        let q = first_true(num_lo, num_hi, |p| !in_band_at(p));
        (q > num_lo).then_some(q as i64 - 1)
    }
}

/// The numeric value of the FIRST `ORDER BY` key at ordered position `p`, or `None` when
/// it is non-numeric/absent (NULL/TEXT/BLOB). The RANGE band is measured against this.
fn numeric_at(part: &Partition, p: usize) -> Option<f64> {
    part.order_key(p).first().and_then(numeric_f64)
}

/// The first index in `[lo, hi)` for which `pred` is true, or `hi` if none is — the
/// standard binary "partition point" over an index range whose predicate is monotonic
/// (all-false then all-true). O(log(hi − lo)) probes; `lo == hi` returns `lo`.
fn first_true(lo: usize, hi: usize, pred: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (lo, hi);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// The pre-binary-search LINEAR reference for [`range_bound`] — a verbatim copy of the
/// O(n) full-partition scan, kept ONLY as the equivalence oracle. The
/// `range_bound_binary_search_matches_linear_scan` test asserts the shipping binary-search
/// [`range_bound`] returns byte-identical `Option<i64>` to this over an exhaustive matrix,
/// so any divergence fails the build. Not compiled outside tests.
#[cfg(test)]
fn range_bound_linear(
    part: &Partition,
    c: usize,
    e: &Value,
    following: bool,
    desc: bool,
    is_start: bool,
) -> Option<i64> {
    let xc = match part.order_key(c).first().and_then(numeric_f64) {
        Some(x) => x,
        None => {
            let (lo, hi) = part.peer_bounds(c);
            return Some(if is_start { lo as i64 } else { hi as i64 - 1 });
        }
    };
    let off = if following { numeric_f64(e).unwrap_or(0.0) } else { -numeric_f64(e).unwrap_or(0.0) };
    let threshold = if desc { xc - off } else { xc + off };

    let in_band = |xi: f64| -> bool {
        match (desc, is_start) {
            (false, true) => xi >= threshold,
            (false, false) => xi <= threshold,
            (true, true) => xi <= threshold,
            (true, false) => xi >= threshold,
        }
    };

    if is_start {
        for p in 0..part.len() {
            if let Some(xi) = part.order_key(p).first().and_then(numeric_f64) {
                if in_band(xi) {
                    return Some(p as i64);
                }
            }
        }
        None
    } else {
        let mut found = None;
        for p in 0..part.len() {
            if let Some(xi) = part.order_key(p).first().and_then(numeric_f64) {
                if in_band(xi) {
                    found = Some(p as i64);
                }
            }
        }
        found
    }
}

/// A `ROWS`/`GROUPS` offset as a signed count (`FOLLOWING` positive). The value has been
/// validated non-negative integer by the shell; this reads it back defensively. A REAL is
/// converted with `as i64`, which truncates toward zero WITHIN range and SATURATES to the
/// i64 extreme for a huge magnitude (so `1e19` reads as `i64::MAX`); a non-numeric value
/// reads as 0. The negation SATURATES so a `PRECEDING` offset near `i64::MAX` cannot
/// overflow — every caller feeds the result into `saturating_add`, and an offset past the
/// partition length then clamps at the frame's `[0, len-1]` intersection.
fn signed_rows(e: &Value, following: bool) -> i64 {
    let n = match e {
        Value::Integer(i) => *i,
        Value::Real(r) => *r as i64,
        _ => 0,
    };
    if following {
        n
    } else {
        n.saturating_neg()
    }
}

/// The numeric value of `v` as `f64` (INTEGER/REAL only), else `None`.
fn numeric_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(i) => Some(*i as f64),
        Value::Real(r) => Some(*r),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_types::Collation;

    fn key(desc: bool) -> SortKey {
        SortKey {
            expr: minisqlite_expr::EvalExpr::Column(0),
            desc,
            nulls_first: None,
            collation: Collation::Binary,
        }
    }

    /// Build a partition whose ordered rows carry the given single INTEGER order key
    /// each (already in window order). Input index == ordered position for simplicity.
    fn part_int(keys: &[i64]) -> Partition {
        let rows: Vec<(usize, Vec<Value>)> =
            keys.iter().enumerate().map(|(i, &k)| (i, vec![Value::Integer(k)])).collect();
        Partition::new(rows, std::slice::from_ref(&key(false)))
    }

    /// Build a partition with a single TEXT order key each, matching sorted window order.
    fn part_text(keys: &[&str]) -> Partition {
        let rows: Vec<(usize, Vec<Value>)> =
            keys.iter().enumerate().map(|(i, k)| (i, vec![Value::Text((*k).into())])).collect();
        Partition::new(rows, std::slice::from_ref(&key(false)))
    }

    /// Build a partition from arbitrary single order-key values under one key (asc/desc).
    /// `Partition::new` sorts them into window order exactly as the operator would, so the
    /// numeric values land in the contiguous storage-class run `range_bound` relies on —
    /// the realistic shape the equivalence test must cover.
    fn part_vals(vals: &[Value], desc: bool) -> Partition {
        let rows: Vec<(usize, Vec<Value>)> =
            vals.iter().cloned().enumerate().map(|(i, v)| (i, vec![v])).collect();
        Partition::new(rows, std::slice::from_ref(&key(desc)))
    }

    fn f(
        units: FrameUnits,
        start: ResolvedBound,
        end: ResolvedBound,
        exclude: FrameExclude,
    ) -> ResolvedFrame {
        ResolvedFrame { units, start, end, exclude }
    }

    /// Every row's frame under `frame`, as position lists in window order (one per row) —
    /// the shape most tests assert against.
    fn frames(p: &Partition, frame: &ResolvedFrame, keys: &[SortKey]) -> Vec<Vec<usize>> {
        (0..p.len()).map(|c| frame_positions(p, c, frame, keys).positions().collect()).collect()
    }

    /// Concatenate the frame's positions (as labels) with '.', like `group_concat`.
    fn gc(labels: &[&str], part: &Partition, c: usize, frame: &ResolvedFrame) -> String {
        frame_positions(part, c, frame, std::slice::from_ref(&key(false)))
            .positions()
            .map(|p| labels[part.input_index(p)])
            .collect::<Vec<_>>()
            .join(".")
    }

    // ---- windowfunctions.html §2.2.3 EXCLUDE table (GROUPS UNBOUNDED PRECEDING .. CURRENT ROW) ----
    // t1 ORDER BY c ⇒ ordered labels A,D,G (c='one'), C,F ('three'), B,E ('two').
    // Model c as the group key: one=0, three=1, two=2 (their sorted order).
    fn t1_by_c() -> (Partition, Vec<&'static str>) {
        (part_int(&[0, 0, 0, 1, 1, 2, 2]), vec!["A", "D", "G", "C", "F", "B", "E"])
    }

    fn groups_default_end(exclude: FrameExclude) -> ResolvedFrame {
        f(FrameUnits::Groups, ResolvedBound::UnboundedPreceding, ResolvedBound::CurrentRow, exclude)
    }

    #[test]
    fn exclude_no_others_matches_spec_table() {
        let (p, l) = t1_by_c();
        let frame = groups_default_end(FrameExclude::NoOthers);
        let got: Vec<String> = (0..p.len()).map(|c| gc(&l, &p, c, &frame)).collect();
        assert_eq!(
            got,
            vec!["A.D.G", "A.D.G", "A.D.G", "A.D.G.C.F", "A.D.G.C.F", "A.D.G.C.F.B.E", "A.D.G.C.F.B.E"]
        );
    }

    #[test]
    fn exclude_current_row_matches_spec_table() {
        let (p, l) = t1_by_c();
        let frame = groups_default_end(FrameExclude::CurrentRow);
        let got: Vec<String> = (0..p.len()).map(|c| gc(&l, &p, c, &frame)).collect();
        assert_eq!(got, vec!["D.G", "A.G", "A.D", "A.D.G.F", "A.D.G.C", "A.D.G.C.F.E", "A.D.G.C.F.B"]);
    }

    #[test]
    fn exclude_group_matches_spec_table() {
        let (p, l) = t1_by_c();
        let frame = groups_default_end(FrameExclude::Group);
        let got: Vec<String> = (0..p.len()).map(|c| gc(&l, &p, c, &frame)).collect();
        assert_eq!(got, vec!["", "", "", "A.D.G", "A.D.G", "A.D.G.C.F", "A.D.G.C.F"]);
    }

    #[test]
    fn exclude_ties_matches_spec_table() {
        let (p, l) = t1_by_c();
        let frame = groups_default_end(FrameExclude::Ties);
        let got: Vec<String> = (0..p.len()).map(|c| gc(&l, &p, c, &frame)).collect();
        assert_eq!(got, vec!["A", "D", "G", "A.D.G.C", "A.D.G.F", "A.D.G.C.F.B", "A.D.G.C.F.E"]);
    }

    // ---- windowfunctions.html §2.2.2 ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING ----
    #[test]
    fn rows_current_to_unbounded_following_matches_spec() {
        let (p, l) = t1_by_c(); // ordered labels A,D,G,C,F,B,E
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::CurrentRow,
            ResolvedBound::UnboundedFollowing,
            FrameExclude::NoOthers,
        );
        let got: Vec<String> = (0..p.len()).map(|c| gc(&l, &p, c, &frame)).collect();
        assert_eq!(got, vec!["A.D.G.C.F.B.E", "D.G.C.F.B.E", "G.C.F.B.E", "C.F.B.E", "F.B.E", "B.E", "E"]);
    }

    // ---- windowfunctions.html §2 ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING (unique key) ----
    #[test]
    fn rows_sliding_one_preceding_one_following_matches_spec() {
        let p = part_int(&[1, 2, 3, 4, 5, 6, 7]); // unique keys ⇒ each row its own peer group
        let l = vec!["A", "B", "C", "D", "E", "F", "G"];
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::Preceding(Value::Integer(1)),
            ResolvedBound::Following(Value::Integer(1)),
            FrameExclude::NoOthers,
        );
        let got: Vec<String> = (0..p.len()).map(|c| gc(&l, &p, c, &frame)).collect();
        assert_eq!(got, vec!["A.B", "A.B.C", "B.C.D", "C.D.E", "D.E.F", "E.F.G", "F.G"]);
    }

    #[test]
    fn rows_following_only_is_empty_past_partition_end() {
        let p = part_int(&[0, 1, 2, 3, 4]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::Following(Value::Integer(1)),
            ResolvedBound::Following(Value::Integer(2)),
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![1, 2], vec![2, 3], vec![3, 4], vec![4], Vec::<usize>::new()]);
    }

    #[test]
    fn rows_preceding_only_is_empty_before_partition_start() {
        let p = part_int(&[0, 1, 2, 3, 4]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::Preceding(Value::Integer(2)),
            ResolvedBound::Preceding(Value::Integer(1)),
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        // c=0: end=-1 ⇒ empty. c=1: [.. 0]. c=2:[0,1]. c=3:[1,2]. c=4:[2,3].
        assert_eq!(got, vec![Vec::<usize>::new(), vec![0], vec![0, 1], vec![1, 2], vec![2, 3]]);
    }

    // ---- RANGE ----
    #[test]
    fn range_default_current_row_includes_peers() {
        // keys 5,10,20,20,30 asc; RANGE UNBOUNDED PRECEDING .. CURRENT ROW.
        let p = part_int(&[5, 10, 20, 20, 30]);
        let frame = f(
            FrameUnits::Range,
            ResolvedBound::UnboundedPreceding,
            ResolvedBound::CurrentRow,
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![0], vec![0, 1], vec![0, 1, 2, 3], vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn range_numeric_band_one_preceding_one_following() {
        let p = part_int(&[1, 2, 3, 10]);
        let frame = f(
            FrameUnits::Range,
            ResolvedBound::Preceding(Value::Integer(1)),
            ResolvedBound::Following(Value::Integer(1)),
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![0, 1], vec![0, 1, 2], vec![1, 2], vec![3]]);
    }

    #[test]
    fn range_desc_band() {
        // DESC: keys descend in window order. 30,20,20,10,5. RANGE 1 PRECEDING .. 1 FOLLOWING.
        let rows: Vec<(usize, Vec<Value>)> =
            [30, 20, 20, 10, 5].iter().enumerate().map(|(i, &k)| (i, vec![Value::Integer(k)])).collect();
        let p = Partition::new(rows, &[key(true)]);
        let frame = f(
            FrameUnits::Range,
            ResolvedBound::Preceding(Value::Integer(1)),
            ResolvedBound::Following(Value::Integer(1)),
            FrameExclude::NoOthers,
        );
        // c at v=20 (pos1,2): band [19,21] ⇒ the two 20s. c at v=30 (pos0): band[29,31] ⇒ pos0.
        let got = frames(&p, &frame, &[key(true)]);
        assert_eq!(got, vec![vec![0], vec![1, 2], vec![1, 2], vec![3], vec![4]]);
    }

    #[test]
    fn range_non_numeric_current_value_uses_peers() {
        // TEXT order values ⇒ non-numeric ⇒ RANGE offset behaves as CURRENT ROW (peers).
        let p = part_text(&["a", "b", "b", "c"]);
        let frame = f(
            FrameUnits::Range,
            ResolvedBound::UnboundedPreceding,
            ResolvedBound::Following(Value::Integer(1)),
            FrameExclude::NoOthers,
        );
        // end bound Following(1) with non-numeric Xc ⇒ last peer. For the two 'b' (pos1,2): end=2.
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![0], vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2, 3]]);
    }

    #[test]
    fn range_bound_binary_search_matches_linear_scan() {
        // The binary-search `range_bound` MUST return byte-identical `Option<i64>` to the
        // linear reference `range_bound_linear` over an exhaustive matrix of shapes ×
        // ASC/DESC × is_start × following × offsets. This IS the proof the O(log n) rewrite
        // is correctness-preserving — perturb the search and it goes red. Mirrors
        // `navigation_helpers_match_positions`.
        //
        // Each partition is built through `Partition::new`, so its numeric values form the
        // contiguous storage-class run the search relies on; the shapes cover all-numeric
        // ascending & descending (with duplicates/peers), INTEGER/REAL interleaved, NULL &
        // TEXT bracketing the run at both ends (and mixed with numerics before sorting), a
        // non-numeric current value (peer path), and empty & single-row partitions. The
        // offsets make the band threshold land before / inside / after the numeric run.
        use Value::{Integer as Int, Null, Real, Text};
        let cases: Vec<(Vec<Value>, bool)> = vec![
            (vec![], false),                                         // empty: no c to test
            (vec![Int(7)], false),                                  // single numeric
            (vec![Null], false),                                    // single non-numeric (peer)
            (vec![Text("q".into())], true),                         // single non-numeric, desc
            (vec![Int(1), Int(2), Int(3), Int(10)], false),         // strictly ascending
            (vec![Int(1), Int(1), Int(2), Int(2), Int(2), Int(3), Int(5)], false), // dups/peers
            (vec![Real(1.5), Int(2), Real(2.5), Int(3)], false),    // int/real interleaved
            (vec![Int(30), Int(20), Int(20), Int(10), Int(5)], true), // descending peers
            (vec![Null, Int(1), Int(2), Int(3), Text("z".into())], false), // brackets both ends
            (
                vec![Text("m".into()), Int(0), Null, Int(9), Int(0), Text("a".into()), Null],
                false,
            ), // non-numerics mixed in pre-sort ⇒ bracket the run post-sort
            (vec![Text("a".into()), Int(5), Int(6), Null], true),   // desc: text low, null high
            (vec![Null, Text("x".into())], false),                  // all non-numeric
            // >2^53 lossy regime — the subtle case the doc flags as a potential
            // divergence witness. It does NOT diverge: `numeric_f64` is
            // `round_to_f64(exact_value)` (identity on REALs, round-to-nearest on integers),
            // and round-to-nearest is globally monotonic, so the EXACT-sorted numeric run
            // stays f64-monotonic even for interleaved INTEGER/REAL past 2^53 — keeping the
            // partition-point premise valid. `9007199254740992.5` is not f64-representable and
            // stores as `2^53`, so the run is `[2^53, 2^53]`, not a decreasing `[…992.5, …992]`.
            (vec![Real(9007199254740992.5), Int(9007199254740993)], false), // the witness
            (vec![Int(9007199254740993), Int(9007199254740994)], false),    // 2^53+1,+2 (a tie)
            (vec![Int(9007199254740995), Int(9007199254740996)], false),    // 2^53+3,+4 (a tie)
            (vec![Int(i64::MAX - 1), Int(i64::MAX)], false),                 // extreme magnitude
            (vec![Real(9007199254740992.0), Int(9007199254740993), Real(9007199254740994.0)], false),
            (vec![Int(9007199254740996), Real(9007199254740994.0), Int(9007199254740993)], true), // desc mixed
        ];
        // Non-negative offsets (as the shell validates them): 0 lands the threshold on `Xc`;
        // the 2^53 offset drives the band arithmetic through the lossy regime too.
        let offsets = [Int(0), Int(1), Int(2), Int(100), Real(1.5), Int(9007199254740992)];
        for (vals, desc) in &cases {
            let part = part_vals(vals, *desc);
            for c in 0..part.len() {
                for off in &offsets {
                    for following in [false, true] {
                        for is_start in [false, true] {
                            let got = range_bound(&part, c, off, following, *desc, is_start);
                            let want = range_bound_linear(&part, c, off, following, *desc, is_start);
                            assert_eq!(
                                got, want,
                                "vals={vals:?} desc={desc} c={c} off={off:?} \
                                 following={following} is_start={is_start}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn range_bound_correct_at_scale() {
        // A few-thousand-row single ascending partition exercised through `range_bound` for a
        // `1 PRECEDING .. 1 FOLLOWING` band. This checks CORRECTNESS at scale — the bounds
        // stay exact (against the closed form for unique keys) and agree with the linear
        // oracle on a large run, not just toys. It does NOT assert timing/complexity, so it
        // would stay green under an O(n) regression; the sub-quadratic guarantee rests on the
        // binary-search code + the exhaustive equivalence oracle above, not this test.
        let n = 4000usize;
        let vals: Vec<Value> = (0..n as i64).map(Value::Integer).collect();
        let part = part_vals(&vals, false);
        let one = Value::Integer(1);
        for &c in &[0usize, 1, 2, 1000, n / 2, n - 2, n - 1] {
            // START = 1 PRECEDING (threshold Xc−1): first xi ≥ c−1 ⇒ position max(c−1, 0).
            let start = range_bound(&part, c, &one, /* following */ false, false, /* is_start */ true);
            assert_eq!(start, Some(c.saturating_sub(1) as i64), "start c={c}");
            assert_eq!(start, range_bound_linear(&part, c, &one, false, false, true));
            // END = 1 FOLLOWING (threshold Xc+1): last xi ≤ c+1 ⇒ position min(c+1, n−1).
            let end = range_bound(&part, c, &one, /* following */ true, false, /* is_start */ false);
            assert_eq!(end, Some((c + 1).min(n - 1) as i64), "end c={c}");
            assert_eq!(end, range_bound_linear(&part, c, &one, true, false, false));
        }
    }

    // ---- GROUPS offset ----
    #[test]
    fn groups_preceding_following_counts_peer_groups() {
        // keys grouped: [0,0,1,2,2] ⇒ groups (0,2)(2,3)(3,5).
        let p = part_int(&[0, 0, 1, 2, 2]);
        let frame = f(
            FrameUnits::Groups,
            ResolvedBound::Preceding(Value::Integer(1)),
            ResolvedBound::Following(Value::Integer(1)),
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        // c in group0(pos0,1): groups 0..1 ⇒ [0,2]. c in group1(pos2): groups 0..2 ⇒ [0,4].
        // c in group2(pos3,4): groups 1..2 ⇒ [2,4].
        assert_eq!(
            got,
            vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2, 3, 4], vec![2, 3, 4], vec![2, 3, 4]]
        );
    }

    // ---- overflow safety: a huge offset must clamp, never overflow `c + off` ----
    // Under the default dev/test profile overflow-checks are on, so an unchecked add would
    // PANIC; in release it would wrap to a negative and silently widen the frame. These pin
    // the saturating boundary arithmetic at both the ROWS and GROUPS sites.

    #[test]
    fn rows_huge_following_offset_saturates_not_panics() {
        // ROWS BETWEEN CURRENT ROW AND i64::MAX FOLLOWING: clamps to the partition end.
        // c=1 exercises `1 + i64::MAX`, which overflows without saturation.
        let p = part_int(&[1, 2]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::CurrentRow,
            ResolvedBound::Following(Value::Integer(i64::MAX)),
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![0, 1], vec![1]]);
    }

    #[test]
    fn groups_huge_following_offset_saturates_not_panics() {
        // GROUPS variant: `group_index(c) + i64::MAX` must saturate then clamp to the last
        // group. Unique keys ⇒ each row is its own peer group.
        let p = part_int(&[1, 2]);
        let frame = f(
            FrameUnits::Groups,
            ResolvedBound::CurrentRow,
            ResolvedBound::Following(Value::Integer(i64::MAX)),
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![0, 1], vec![1]]);
    }

    #[test]
    fn rows_huge_following_start_bound_saturates_not_panics() {
        // START-bound twin of `rows_huge_following_offset_saturates_not_panics`: a huge
        // FOLLOWING as the START bound routes through `start_offset`'s Rows arm, NOT
        // `end_offset`. At c=1 the start is `1 + i64::MAX`, which panics without saturation;
        // with it the start saturates to i64::MAX and the `[0, len-1]` intersection makes
        // every frame empty (the start is past the partition end) — SQLite's clamping. This
        // is the guard the end-bound tests miss (they exercise only `end_offset`).
        let p = part_int(&[1, 2]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::Following(Value::Integer(i64::MAX)),
            ResolvedBound::UnboundedFollowing,
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![Vec::<usize>::new(), Vec::<usize>::new()]);
    }

    #[test]
    fn groups_huge_following_start_bound_saturates_not_panics() {
        // GROUPS twin at the START bound: `group_index(c) + i64::MAX` saturates in
        // `start_offset`'s Groups arm; the saturated group is past the last group, so the
        // boundary reports "no such row" (None) and the frame is empty. At c=1 the
        // unsaturated add panics — the start_offset Groups guard the end-bound tests miss.
        let p = part_int(&[1, 2]);
        let frame = f(
            FrameUnits::Groups,
            ResolvedBound::Following(Value::Integer(i64::MAX)),
            ResolvedBound::UnboundedFollowing,
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![Vec::<usize>::new(), Vec::<usize>::new()]);
    }

    #[test]
    fn rows_huge_preceding_offset_saturates_not_panics() {
        // i64::MAX PRECEDING clamps the start to the partition front (a running total). This
        // pins the clamp *direction* to [0, c], not saturation itself: for a validated
        // non-negative offset `signed_rows` yields `-i64::MAX` and `c + (-i64::MAX)` stay in
        // range, so it can't overflow either way — the FOLLOWING-start tests above guard the
        // saturating arithmetic; this one guards that PRECEDING clamps low rather than empty.
        let p = part_int(&[1, 2, 3]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::Preceding(Value::Integer(i64::MAX)),
            ResolvedBound::CurrentRow,
            FrameExclude::NoOthers,
        );
        let got = frames(&p, &frame, &[key(false)]);
        assert_eq!(got, vec![vec![0], vec![0, 1], vec![0, 1, 2]]);
    }

    // ---- navigation helpers ----
    #[test]
    fn navigation_first_last_nth() {
        let p = part_int(&[10, 20, 30, 40, 50]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::UnboundedPreceding,
            ResolvedBound::CurrentRow,
            FrameExclude::NoOthers,
        );
        let fr = frame_positions(&p, 3, &frame, &[key(false)]); // frame [0,3]
        assert_eq!(fr.first(), Some(0));
        assert_eq!(fr.last(), Some(3));
        assert_eq!(fr.nth(2), Some(2)); // 0-based 3rd position
        assert_eq!(fr.nth(9), None);
        assert!(!fr.is_empty());
    }

    #[test]
    fn empty_frame_navigation_is_none() {
        let p = part_int(&[1, 2, 3]);
        let frame = f(
            FrameUnits::Rows,
            ResolvedBound::Following(Value::Integer(5)),
            ResolvedBound::Following(Value::Integer(9)),
            FrameExclude::NoOthers,
        );
        let fr = frame_positions(&p, 0, &frame, &[key(false)]);
        assert!(fr.is_empty());
        assert_eq!(fr.first(), None);
        assert_eq!(fr.last(), None);
        assert_eq!(fr.nth(0), None);
    }

    #[test]
    fn navigation_helpers_match_positions() {
        // The O(1) `first`/`last`/`nth`/`is_empty` (built on `segments()`) must agree EXACTLY
        // with the reference `positions()` walk over every frame shape × EXCLUDE mode ×
        // current row. This pins `segments()` to the tested `positions()`/`is_excluded` path,
        // so the fast navigation helpers can never silently diverge from the frame the
        // aggregate fold sees.
        let parts = [
            part_int(&[1, 2, 3, 4, 5]),    // unique keys ⇒ singleton peer groups
            part_int(&[0, 0, 1, 1, 1, 2]), // peer groups of size 2, 3, 1
            part_int(&[7]),                // single row
            part_int(&[4, 4, 4, 4]),       // one big peer group
        ];
        let bounds = [
            (ResolvedBound::UnboundedPreceding, ResolvedBound::CurrentRow),
            (ResolvedBound::CurrentRow, ResolvedBound::UnboundedFollowing),
            (ResolvedBound::UnboundedPreceding, ResolvedBound::UnboundedFollowing),
            (ResolvedBound::Preceding(Value::Integer(1)), ResolvedBound::Following(Value::Integer(1))),
            (ResolvedBound::Following(Value::Integer(1)), ResolvedBound::Following(Value::Integer(2))),
            (ResolvedBound::Preceding(Value::Integer(2)), ResolvedBound::Preceding(Value::Integer(1))),
        ];
        let excludes =
            [FrameExclude::NoOthers, FrameExclude::CurrentRow, FrameExclude::Group, FrameExclude::Ties];
        let k = [key(false)];
        for p in &parts {
            for units in [FrameUnits::Rows, FrameUnits::Range, FrameUnits::Groups] {
                for (start, end) in &bounds {
                    for exclude in excludes {
                        let frame = f(units, start.clone(), end.clone(), exclude);
                        for c in 0..p.len() {
                            let fr = frame_positions(p, c, &frame, &k);
                            let want: Vec<usize> = fr.positions().collect();
                            let ctx = format!("{units:?} {start:?}..{end:?} {exclude:?} c={c}");
                            assert_eq!(fr.first(), want.first().copied(), "first {ctx}");
                            assert_eq!(fr.last(), want.last().copied(), "last {ctx}");
                            assert_eq!(fr.is_empty(), want.is_empty(), "is_empty {ctx}");
                            // Probe every valid index plus one past the end (⇒ `None`).
                            for i in 0..=want.len() + 1 {
                                assert_eq!(fr.nth(i), want.get(i).copied(), "nth({i}) {ctx}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn is_default_detects_default_frame_only() {
        use FrameExclude::{CurrentRow as ExcCurrent, NoOthers};
        use FrameUnits::{Range, Rows};
        use ResolvedBound::{CurrentRow, UnboundedFollowing, UnboundedPreceding};
        // The default is exactly RANGE UNBOUNDED PRECEDING .. CURRENT ROW EXCLUDE NO OTHERS.
        assert!(f(Range, UnboundedPreceding, CurrentRow, NoOthers).is_default());
        // Wrong units, wrong exclude, or wrong end bound each fail the match.
        assert!(!f(Rows, UnboundedPreceding, CurrentRow, NoOthers).is_default());
        assert!(!f(Range, UnboundedPreceding, CurrentRow, ExcCurrent).is_default());
        assert!(!f(Range, UnboundedPreceding, UnboundedFollowing, NoOthers).is_default());
    }
}
