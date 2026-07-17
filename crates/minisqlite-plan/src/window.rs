//! Window-function vocabulary: the KIND of window function to evaluate and the
//! FRAME it evaluates over. Data only — the binder fills these in from an `OVER`
//! clause, the planner emits a [`crate::Window`] node carrying them, and the
//! executor computes them; there is no behavior here.
//!
//! This is the shared shape three sibling seams bind against (window binding, window
//! plan-emission, window execution), so it must express EVERY SQLite window function,
//! not just aggregate windows. The vocabulary follows `spec/sqlite-doc/windowfunctions.html`:
//! [`WindowFuncKind`] the 11 built-ins of §3 plus the aggregate case, and
//! [`WindowFrame`] the frame specification of §2.2 (`ROWS`/`RANGE`/`GROUPS`, the five
//! boundary forms, and `EXCLUDE`).
//!
//! # No `PartialEq`
//!
//! Every type that embeds an [`EvalExpr`] (or an [`AggregateCall`], which itself embeds
//! `EvalExpr`) is `Debug + Clone` but NOT `PartialEq`: a resolved function handle behind
//! `dyn` is not comparable and structural expression equality is not a concept the plan
//! needs (see `minisqlite_expr`'s `EvalExpr`). The fieldless [`FrameUnits`] and
//! [`FrameExclude`] tags are the exception — they are `Copy + PartialEq + Eq`.

use minisqlite_expr::{AggregateCall, EvalExpr};

/// Which window function a [`crate::WindowFunc`] invokes. Covers the aggregate case
/// (any aggregate used with `OVER`, evaluated over the frame) plus the 11 built-in
/// window functions of `windowfunctions.html` §3.
///
/// The offset/value functions carry their argument expressions directly (bound against
/// the pre-window input row) rather than reusing [`AggregateCall`], because they are not
/// aggregates: `lag`/`lead` read a sibling row, and `first_value`/`last_value`/`nth_value`
/// read a specific row of the frame.
#[derive(Debug, Clone)]
pub enum WindowFuncKind {
    /// An aggregate evaluated over the window frame (`sum`/`count`/`avg`/`min`/`max`/
    /// `group_concat`/… with `OVER`). The [`AggregateCall`] carries the arguments and
    /// `FILTER`; the frame it folds over is the enclosing [`WindowFrame`].
    Aggregate(AggregateCall),
    /// `row_number()` — the 1-based position of the row within its partition, in
    /// `ORDER BY` order (arbitrary order if there is no `ORDER BY`).
    RowNumber,
    /// `rank()` — the `row_number()` of the first peer in the current group: the rank
    /// with gaps. With no `ORDER BY` all rows are peers and this is always 1.
    Rank,
    /// `dense_rank()` — the 1-based number of the current row's peer group within the
    /// partition: the rank without gaps. With no `ORDER BY` this is always 1.
    DenseRank,
    /// `percent_rank()` — `(rank - 1) / (partition_rows - 1)`, in `[0.0, 1.0]`; `0.0`
    /// for a single-row partition.
    PercentRank,
    /// `cume_dist()` — the cumulative distribution `row_number / partition_rows`, where
    /// `row_number` is that of the last peer in the current group.
    CumeDist,
    /// `ntile(N)` — divides the partition into `N` groups as evenly as possible (larger
    /// groups first) and returns the 1..=`N` group the current row falls in. The
    /// argument `N` is evaluated (and read as an integer) against the input row.
    Ntile(EvalExpr),
    /// `lag(expr[, offset[, default]])` — `expr` evaluated against the row `offset` rows
    /// BEFORE the current row in the partition; `default` if that row does not exist.
    /// The frame is ignored. Missing `offset`/`default` are lowered by the binder to the
    /// SQLite defaults (`offset = 1`, `default = NULL`), so all three are always present.
    Lag { expr: EvalExpr, offset: EvalExpr, default: EvalExpr },
    /// `lead(expr[, offset[, default]])` — `expr` evaluated against the row `offset` rows
    /// AFTER the current row in the partition; `default` if that row does not exist. The
    /// frame is ignored; `offset`/`default` default to `1`/`NULL` as for [`Lag`](Self::Lag).
    Lead { expr: EvalExpr, offset: EvalExpr, default: EvalExpr },
    /// `first_value(expr)` — `expr` evaluated against the first row of the window frame.
    FirstValue(EvalExpr),
    /// `last_value(expr)` — `expr` evaluated against the last row of the window frame.
    LastValue(EvalExpr),
    /// `nth_value(expr, N)` — `expr` evaluated against row `N` (1-based) of the window
    /// frame, or NULL if the frame has fewer than `N` rows.
    NthValue { expr: EvalExpr, n: EvalExpr },
}

/// The frame TYPE, which decides how the [`FrameBound`]s are measured
/// (`windowfunctions.html` §2.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameUnits {
    /// `ROWS` — boundaries count individual rows relative to the current row.
    Rows,
    /// `RANGE` — boundaries frame the rows whose `ORDER BY` value is within a band of the
    /// current row's value (requires exactly one `ORDER BY` term for an `<expr>` bound).
    Range,
    /// `GROUPS` — boundaries count peer groups (rows equal on every `ORDER BY` term)
    /// relative to the current group.
    Groups,
}

/// One frame boundary — the start or the end of the frame (`windowfunctions.html`
/// §2.2.2). The `<expr>` of `Preceding`/`Following` is a non-negative constant offset
/// whose meaning (rows / range band / groups) is set by the enclosing [`FrameUnits`].
#[derive(Debug, Clone)]
pub enum FrameBound {
    /// `UNBOUNDED PRECEDING` — the first row of the partition (only valid as a start).
    UnboundedPreceding,
    /// `<expr> PRECEDING` — `<expr>` units before the current row.
    Preceding(EvalExpr),
    /// `CURRENT ROW` — the current row (RANGE/GROUPS: the current row's whole peer group).
    CurrentRow,
    /// `<expr> FOLLOWING` — `<expr>` units after the current row.
    Following(EvalExpr),
    /// `UNBOUNDED FOLLOWING` — the last row of the partition (only valid as an end).
    UnboundedFollowing,
}

/// The `EXCLUDE` clause of a frame: which rows around the current row are dropped from
/// the frame after it is otherwise computed (`windowfunctions.html` §2.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameExclude {
    /// `EXCLUDE NO OTHERS` — the default; exclude nothing.
    NoOthers,
    /// `EXCLUDE CURRENT ROW` — drop the current row (its peers stay).
    CurrentRow,
    /// `EXCLUDE GROUP` — drop the current row and all its peers.
    Group,
    /// `EXCLUDE TIES` — drop the current row's peers but keep the current row itself.
    Ties,
}

/// A resolved window frame specification (`windowfunctions.html` §2.2): the frame type,
/// its start and end boundaries, and the `EXCLUDE` rule. The default (when a window has
/// no explicit `frame-spec`) is [`WindowFrame::default_frame`].
#[derive(Debug, Clone)]
pub struct WindowFrame {
    /// The frame type — how `start`/`end` are measured.
    pub units: FrameUnits,
    /// The start boundary (the frame's lower edge).
    pub start: FrameBound,
    /// The end boundary (the frame's upper edge).
    pub end: FrameBound,
    /// Rows to exclude from the otherwise-computed frame.
    pub exclude: FrameExclude,
}

impl WindowFrame {
    /// The SQLite default frame when a window carries no explicit `frame-spec`
    /// (`windowfunctions.html` §2.2): `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    /// EXCLUDE NO OTHERS` — every row from the start of the partition up to and including
    /// the current row and its peers.
    pub fn default_frame() -> WindowFrame {
        WindowFrame {
            units: FrameUnits::Range,
            start: FrameBound::UnboundedPreceding,
            end: FrameBound::CurrentRow,
            exclude: FrameExclude::NoOthers,
        }
    }
}

// Compile-time contract: the window vocabulary must stay `Send + Sync` so a compiled
// plan carrying it can be shared across WAL reader threads (mirrors the guards in
// `minisqlite_expr`). It holds today because every field decomposes to `EvalExpr` /
// `AggregateCall` (both `Send + Sync`) plus `Copy` tag enums. A sibling filling in
// window binding/execution that embeds a non-`Sync` field (an `Rc`, a `Cell`, a
// `RefCell`) into a `WindowFuncKind` / `WindowFrame` would silently break cross-thread
// plan sharing — this turns that regression into a compile error at the definition
// instead. The closure is never called; coercing it to `fn()` forces its body (and thus
// the bounds) to type-check.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WindowFuncKind>();
    assert_send_sync::<WindowFrame>();
    assert_send_sync::<FrameBound>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_types::Value;

    fn lit(n: i64) -> EvalExpr {
        EvalExpr::Literal(Value::Integer(n))
    }

    /// The default frame is exactly SQLite's `RANGE UNBOUNDED PRECEDING .. CURRENT ROW
    /// EXCLUDE NO OTHERS` (windowfunctions.html §2.2). Pinned so the binder and executor
    /// share one notion of "no explicit frame".
    #[test]
    fn default_frame_is_range_unbounded_preceding_to_current_row() {
        let f = WindowFrame::default_frame();
        assert_eq!(f.units, FrameUnits::Range);
        assert!(matches!(f.start, FrameBound::UnboundedPreceding));
        assert!(matches!(f.end, FrameBound::CurrentRow));
        assert_eq!(f.exclude, FrameExclude::NoOthers);
        // Debug + Clone are wired end to end.
        assert!(!format!("{:?}", f.clone()).is_empty());
    }

    /// Every `WindowFuncKind` variant is constructible and `Debug + Clone` (the shape
    /// sibling tasks bind against). The `Aggregate` case needs an `AggregateCall` handle
    /// and is exercised through the binder (`compile::window`); here we cover the 11
    /// built-ins, whose fields are only `EvalExpr`.
    #[test]
    fn every_builtin_window_func_kind_is_constructible_and_debuggable() {
        let kinds = vec![
            WindowFuncKind::RowNumber,
            WindowFuncKind::Rank,
            WindowFuncKind::DenseRank,
            WindowFuncKind::PercentRank,
            WindowFuncKind::CumeDist,
            WindowFuncKind::Ntile(lit(4)),
            WindowFuncKind::Lag { expr: EvalExpr::Column(0), offset: lit(1), default: EvalExpr::Literal(Value::Null) },
            WindowFuncKind::Lead { expr: EvalExpr::Column(0), offset: lit(2), default: lit(0) },
            WindowFuncKind::FirstValue(EvalExpr::Column(1)),
            WindowFuncKind::LastValue(EvalExpr::Column(1)),
            WindowFuncKind::NthValue { expr: EvalExpr::Column(1), n: lit(3) },
        ];
        for k in &kinds {
            assert!(!format!("{:?}", k.clone()).is_empty());
        }
        assert_eq!(kinds.len(), 11, "the 11 built-in window functions of §3");
    }

    /// A fully explicit frame round-trips through `Debug`/`Clone` and the fieldless tags
    /// compare by value — the executor reads `units`/`exclude` as plain data.
    #[test]
    fn explicit_frame_is_constructible_and_tags_compare_by_value() {
        let f = WindowFrame {
            units: FrameUnits::Rows,
            start: FrameBound::Preceding(lit(2)),
            end: FrameBound::Following(lit(1)),
            exclude: FrameExclude::Ties,
        };
        assert_ne!(f.units, FrameUnits::Range);
        assert_ne!(f.exclude, FrameExclude::NoOthers);
        assert!(matches!(f.start, FrameBound::Preceding(_)));
        assert!(matches!(f.end, FrameBound::Following(_)));
        assert!(!format!("{:?}", f.clone()).is_empty());
        // `Copy` on the tag enums: reading one leaves it usable.
        let u = f.units;
        assert_eq!(u, f.units);
        assert!(matches!(FrameBound::UnboundedFollowing, FrameBound::UnboundedFollowing));
    }
}
