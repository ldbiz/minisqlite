//! NAVIGATION window functions (windowfunctions.html §3): the two OFFSET functions
//! `lag`/`lead` and the three VALUE functions `first_value`/`last_value`/`nth_value`.
//!
//! These bind against the same [`super::partition`]/[`super::frame`] seam the aggregate
//! kind uses, so partitioning, `ORDER BY`, peer groups, and framing are defined ONCE in
//! the shell and reused here rather than reinvented. The split between the two families
//! is the spec's:
//!
//! * `lag`/`lead` IGNORE the frame — they read a sibling row a constant `offset` away in
//!   `ORDER BY` order within the partition, falling back to `default` (evaluated against
//!   the CURRENT row) when that sibling is outside the partition.
//! * `first_value`/`last_value`/`nth_value` RESPECT the frame — they read the first /
//!   last / N-th row of each row's computed window frame (in window order, post-`EXCLUDE`),
//!   yielding NULL when the frame has no such row.
//!
//! Worked oracles (spec §3, `t1.b` = A,B,C,D,E,F,G, all under `WINDOW (ORDER BY b …)`):
//! * `lag(b)`          = NULL,A,B,C,D,E,F        — offset 1; the first row has no prior.
//! * `lead(b,2,'n/a')` = C,D,E,F,G,'n/a','n/a'   — offset 2; past the end ⇒ `default`.
//!   and over `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`:
//! * `first_value(b)`  = A,A,A,A,A,A,A
//! * `last_value(b)`   = A,B,C,D,E,F,G           — the frame grows one row per position.
//! * `nth_value(b,3)`  = NULL,NULL,C,C,C,C,C     — frame-relative, 1-based `N`.

use minisqlite_expr::{eval, to_integer, EvalExpr};
use minisqlite_plan::WindowFunc;
use minisqlite_types::{Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::runtime::Runtime;

/// Which VALUE function [`value_fn`] computes. `Nth` carries the (still unevaluated) `N`
/// argument expression; `First`/`Last` need only the shared `expr`.
#[derive(Clone, Copy)]
pub(super) enum ValueKind<'a> {
    First,
    Last,
    Nth(&'a EvalExpr),
}

/// `lag` (`lead == false`) / `lead` (`lead == true`): for each row, evaluate `expr`
/// against the row `offset` positions BEFORE (`lag`) or AFTER (`lead`) it in `ORDER BY`
/// order within the partition; when that row falls outside the partition, evaluate
/// `default` against the CURRENT row instead. The frame is IGNORED (spec §3).
///
/// `offset` is a constant (spec: "a non-negative integer"): it is evaluated ONCE against
/// an empty outer and coerced with the engine's standard value→integer rule
/// ([`to_integer`] — a real truncates, a text/blob takes its leading integer prefix, NULL
/// is 0). A negative offset is the SQL error real SQLite reports; `offset` 0 targets the
/// current row (`pos - 0 == pos`).
pub(super) fn lag_lead(
    env: Env<'_>,
    rt: &mut Runtime,
    wf: &WindowFunc,
    expr: &EvalExpr,
    offset: &EvalExpr,
    default: &EvalExpr,
    lead: bool,
    all_rows: &[Row],
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    let offset = to_integer(&eval_const(offset, env, rt)?);
    if offset < 0 {
        // Real SQLite's message for a negative lag/lead offset (shared with frame bounds).
        return Err(Error::sql("frame starting offset must be a non-negative integer"));
    }

    let partitions = super::partition_rows(env, rt, wf, all_rows)?;
    debug_assert_eq!(
        partitions.iter().map(|p| p.len()).sum::<usize>(),
        all_rows.len(),
        "window partitions must cover every input row exactly once",
    );

    for part in &partitions {
        let len = part.len() as i64;
        for pos in 0..part.len() {
            // Saturating add on the `lead` side: `offset` is validated >= 0 but not bounded
            // above, so a huge (valid) offset near i64::MAX would overflow `pos + offset`
            // (a panic under overflow-checks). Saturating to i64::MAX lands outside the small
            // `0..len`, so the row falls through to `default` (NULL) — the answer real SQLite
            // gives for an out-of-range offset. Mirrors the frame.rs boundary arithmetic.
            // The `lag` subtract is deliberately left as a plain `-`: it cannot underflow
            // (`pos >= 0`, `offset <= i64::MAX`, so the minimum is `0 - i64::MAX = -i64::MAX`,
            // still `> i64::MIN`), and a saturating subtract there would be dead defense.
            let target =
                if lead { (pos as i64).saturating_add(offset) } else { pos as i64 - offset };
            let value = if (0..len).contains(&target) {
                eval_at(expr, all_rows, part.input_index(target as usize), env, rt)?
            } else {
                // No such sibling row: `default` (which may read the current row's columns).
                eval_at(default, all_rows, part.input_index(pos), env, rt)?
            };
            results[part.input_index(pos)][j] = value;
        }
    }
    Ok(())
}

/// `first_value`/`last_value`/`nth_value`: for each row, compute its window frame (the
/// same frame the aggregate kind folds over) and evaluate `expr` against the first / last
/// / N-th row of that frame in window order (post-`EXCLUDE`), or NULL when the frame has
/// no such row. Unlike `lag`/`lead`, these RESPECT the frame (spec §3).
///
/// For `nth_value`, `N` is a constant 1-based index evaluated ONCE; it must be a positive
/// integer (the SQL error real SQLite reports otherwise). [`super::frame::Frame::nth`] is
/// 0-based, so it is queried with `N - 1`.
pub(super) fn value_fn(
    env: Env<'_>,
    rt: &mut Runtime,
    wf: &WindowFunc,
    expr: &EvalExpr,
    kind: ValueKind<'_>,
    all_rows: &[Row],
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    // Resolve `N` (if any) to a 0-based frame index up front — it is constant, so it is
    // evaluated and validated once for the whole function rather than per row.
    let pick = match kind {
        ValueKind::First => Pick::First,
        ValueKind::Last => Pick::Last,
        ValueKind::Nth(n_expr) => {
            let n = to_integer(&eval_const(n_expr, env, rt)?);
            if n < 1 {
                return Err(Error::sql("argument of nth_value must be a positive integer"));
            }
            Pick::Nth((n - 1) as usize)
        }
    };

    let resolved = super::resolve_frame(&wf.frame, env, rt)?;
    let partitions = super::partition_rows(env, rt, wf, all_rows)?;
    debug_assert_eq!(
        partitions.iter().map(|p| p.len()).sum::<usize>(),
        all_rows.len(),
        "window partitions must cover every input row exactly once",
    );

    for part in &partitions {
        for c in 0..part.len() {
            let fr = super::frame::frame_positions(part, c, &resolved, &wf.order_by);
            let picked = match pick {
                Pick::First => fr.first(),
                Pick::Last => fr.last(),
                Pick::Nth(k) => fr.nth(k),
            };
            let value = match picked {
                Some(p) => eval_at(expr, all_rows, part.input_index(p), env, rt)?,
                None => Value::Null,
            };
            results[part.input_index(c)][j] = value;
        }
    }
    Ok(())
}

/// The VALUE-function selector after `nth_value`'s `N` has been resolved to a 0-based
/// frame index, so the per-row loop reads a plain index instead of re-evaluating `N`.
#[derive(Clone, Copy)]
enum Pick {
    First,
    Last,
    Nth(usize),
}

/// Evaluate a CONSTANT window argument (`lag`/`lead` offset, `nth_value` N). A constant
/// argument carries no column reference, so it is evaluated against an EMPTY outer —
/// exactly as the shell evaluates the frame-bound offsets ([`super::NO_OUTER`]).
fn eval_const(expr: &EvalExpr, env: Env<'_>, rt: &mut Runtime) -> Result<Value> {
    let mut ctx = EvalCtx { rt, env, outer: super::NO_OUTER };
    eval(expr, super::NO_OUTER, &mut ctx)
}

/// Evaluate `expr` against input row `idx` — the row a navigation function reads. Mirrors
/// every operator eval site: the row is both the evaluation registers and the `outer` a
/// correlated subquery inside `expr` would read.
fn eval_at(
    expr: &EvalExpr,
    all_rows: &[Row],
    idx: usize,
    env: Env<'_>,
    rt: &mut Runtime,
) -> Result<Value> {
    let mut ctx = EvalCtx { rt, env, outer: &all_rows[idx] };
    eval(expr, &all_rows[idx], &mut ctx)
}
