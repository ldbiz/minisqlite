//! The RANKING window functions (`windowfunctions.html` §3): `row_number`, `rank`,
//! `dense_rank`, `percent_rank`, `cume_dist`, `ntile`.
//!
//! Ranking is a pure function of the ORDERED partition and its PEER GROUPS — it never
//! reads the frame (`wf.frame` is untouched). So every kind here is computed straight
//! off the [`Partition`](super::partition::Partition) the window CORE already builds:
//! this module reuses [`super::partition_rows`] to group + order the rows, then for each
//! ordered position writes one value. Nothing is materialized beyond the partition
//! machinery the core already owns.
//!
//! # Semantics (each traced against the spec §3 worked example)
//!
//! * `row_number` — 1-based position in the partition: `pos + 1`.
//! * `rank` — the `row_number` of the FIRST peer in the current group (`rank` WITH gaps):
//!   `peer_bounds(pos).0 + 1`. No `ORDER BY` ⇒ one peer group ⇒ always 1.
//! * `dense_rank` — the 1-based peer-group ordinal (`rank` WITHOUT gaps):
//!   `group_index(pos) + 1`. No `ORDER BY` ⇒ always 1.
//! * `percent_rank` — `(rank - 1) / (rows - 1)`, or `0.0` for a single-row partition.
//! * `cume_dist` — `(row_number of the LAST peer) / rows` = `peer_bounds(pos).1 / rows`
//!   (the half-open peer upper bound is the count of rows through the group's end).
//! * `ntile(N)` — split the partition into `N` buckets as evenly as possible, LARGER
//!   buckets first, numbered `1..=N`. `N` is evaluated once (constant) and must be a
//!   positive integer.

use minisqlite_expr::{eval, to_integer, EvalExpr};
use minisqlite_plan::WindowFunc;
use minisqlite_types::{Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::runtime::Runtime;

use super::partition::Partition;

/// Which ranking function to compute. Five kinds are fieldless; `Ntile` carries a
/// borrow of its argument expression (a constant `N`). `Copy` so the per-position loop
/// in [`compute`] can re-read it without moving. Passed in by the dispatch in the
/// parent `window` module so each ranking kind flows through the one `compute` entry.
#[derive(Clone, Copy)]
pub(crate) enum RankingKind<'a> {
    RowNumber,
    Rank,
    DenseRank,
    PercentRank,
    CumeDist,
    Ntile(&'a EvalExpr),
}

/// Compute one ranking function's appended value for every input row, writing
/// `results[input_index(pos)][j]`.
///
/// Partitions the input ONCE via [`super::partition_rows`] (shared with the aggregate
/// path, so the partition/peer definition cannot drift), then walks each partition in
/// window order. For `ntile`, the argument is evaluated + validated ONCE up front (it
/// is a constant), so a bad `N` is a single loud error regardless of input size —
/// mirroring how the aggregate path resolves frame bounds before partitioning.
pub(crate) fn compute(
    env: Env<'_>,
    rt: &mut Runtime,
    wf: &WindowFunc,
    kind: RankingKind<'_>,
    all_rows: &[Row],
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    // Resolve ntile's `N` before partitioning: it is constant, and an invalid `N` must
    // fail once, not per partition/row.
    let ntile_n = match kind {
        RankingKind::Ntile(e) => Some(eval_ntile_n(e, env, rt)?),
        _ => None,
    };

    let partitions = super::partition_rows(env, rt, wf, all_rows)?;

    // The partitions cover `0..all_rows.len()` exactly once (same invariant the
    // aggregate path relies on), so every `results[i][j]` placeholder is overwritten
    // and no `Value::Null` placeholder can survive as a real ranking value.
    debug_assert_eq!(
        partitions.iter().map(Partition::len).sum::<usize>(),
        all_rows.len(),
        "window partitions must cover every input row exactly once",
    );

    for part in &partitions {
        let len = part.len();
        for pos in 0..len {
            let value = match kind {
                RankingKind::RowNumber => Value::Integer((pos + 1) as i64),
                RankingKind::Rank => Value::Integer((part.peer_bounds(pos).0 + 1) as i64),
                RankingKind::DenseRank => Value::Integer((part.group_index(pos) + 1) as i64),
                RankingKind::PercentRank => Value::Real(percent_rank(part, pos)),
                RankingKind::CumeDist => Value::Real(cume_dist(part, pos)),
                RankingKind::Ntile(_) => {
                    let n = ntile_n.expect("ntile N is resolved for the Ntile kind");
                    Value::Integer(ntile_bucket(len, pos, n) as i64)
                }
            };
            results[part.input_index(pos)][j] = value;
        }
    }
    Ok(())
}

/// `percent_rank()` = `(rank - 1) / (rows - 1)`, defined as `0.0` for a single-row (or
/// empty) partition. `rank` is `peer_bounds(pos).0 + 1`, so `rank - 1` is exactly the
/// peer group's ordered start index — no `+1`/`-1` round trip.
fn percent_rank(part: &Partition, pos: usize) -> f64 {
    let len = part.len();
    if len <= 1 {
        return 0.0;
    }
    let rank_minus_one = part.peer_bounds(pos).0;
    rank_minus_one as f64 / (len - 1) as f64
}

/// `cume_dist()` = (row_number of the last peer in the group) / partition rows. The
/// peer group's half-open upper bound `peer_bounds(pos).1` is precisely that count (the
/// number of rows from the partition start through the end of the current peer group).
fn cume_dist(part: &Partition, pos: usize) -> f64 {
    part.peer_bounds(pos).1 as f64 / part.len() as f64
}

/// The 1-based `ntile(n)` bucket for ordered position `pos` in a partition of `len`
/// rows: divide `len` rows into `n` buckets as evenly as possible, LARGER buckets first.
/// The first `len % n` buckets hold `len/n + 1` rows; the remaining buckets hold `len/n`.
///
/// Preconditions (guaranteed by the caller): `n >= 1` (validated in [`eval_ntile_n`])
/// and `pos < len` (the loop bound). When `n > len` the smaller bucket size is 0 and the
/// larger buckets tile the whole partition (`rem == len`, `large == len`), so every `pos`
/// takes the first branch and the second branch's `/ base` is never reached with
/// `base == 0` — the first `len` buckets get one row each and buckets `len+1..=n` are
/// empty, matching SQLite.
fn ntile_bucket(len: usize, pos: usize, n: i64) -> usize {
    debug_assert!(n >= 1, "ntile N is validated positive before bucketing");
    debug_assert!(pos < len, "ntile position must be within the partition");
    // `n >= 1`, and on a 64-bit target `usize` holds every positive `i64`; `.max(1)`
    // is a belt-and-braces guard so a (32-bit only) truncation to 0 cannot divide by 0.
    let n = (n as usize).max(1);
    let base = len / n; // the smaller bucket size
    let rem = len % n; // this many leading buckets are one row larger
    let large = rem * (base + 1); // ordered positions covered by the larger buckets
    if pos < large {
        pos / (base + 1) + 1
    } else {
        rem + (pos - large) / base + 1
    }
}

/// Evaluate `ntile(N)`'s constant argument once (against an empty outer, like the
/// aggregate path's frame-bound eval) and coerce it to an integer with the engine's
/// shared value→integer rule ([`to_integer`] — NULL → 0, REAL truncates toward zero,
/// TEXT/BLOB take the longest leading integer prefix), the same reader `sqlite3_value_int64`
/// uses. SQLite requires `N` to be a positive integer, so any `N < 1` — including the `0`
/// a NULL argument coerces to — is a loud SQL error, never a panic.
fn eval_ntile_n(e: &EvalExpr, env: Env<'_>, rt: &mut Runtime) -> Result<i64> {
    let v = {
        let mut ctx = EvalCtx { rt, env, outer: super::NO_OUTER };
        eval(e, super::NO_OUTER, &mut ctx)?
    };
    let n = to_integer(&v);
    if n < 1 {
        return Err(Error::sql("argument of ntile must be a positive integer"));
    }
    Ok(n)
}
