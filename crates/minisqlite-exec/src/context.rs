//! [`EvalCtx`] — the executor's implementation of the expression evaluator's
//! [`EvalContext`] / [`FnContext`] seams.
//!
//! The evaluator ([`minisqlite_expr::eval`]) reaches out through these traits for
//! everything it cannot compute from the row alone: the wall clock, the RNG, the
//! connection counters, bound parameters, and the subquery callbacks. `EvalCtx`
//! wires those to the [`Runtime`] (RNG + counters + params) and the [`Env`] (the
//! plan + catalog + pager a subquery would need).
//!
//! An `EvalCtx` is constructed fresh at each operator eval site over the current
//! row, e.g. `EvalCtx { rt, env: self.env, outer: &row }` — the current row is both
//! the evaluation registers and the `outer` a nested (correlated) subquery would
//! read. The subquery callbacks stream a subplan by resolving
//! `env.plan.subqueries[id]` and running it through
//! [`build_cursor`](crate::runner::build_cursor), handing a correlated subplan the
//! current row as its outer.

use std::cmp::Ordering;
use std::collections::HashSet;

use minisqlite_expr::{CompareMeta, EvalContext, FnContext, SubqueryId};
use minisqlite_plan::{Plan, SubPlan};
use minisqlite_types::{apply_affinity, compare_for_eq, Affinity, Error, Result, Row, Value};

use crate::corr_key::{corr_key, CorrCell};
use crate::env::Env;
use crate::keys::{cell_key, CellKey};
use crate::runner::build_cursor;
use crate::runtime::{CachedSubquery, Runtime};
use crate::RowCursor;

/// The evaluation context handed to `minisqlite_expr::eval` at every operator eval
/// site. Borrows the connection [`Runtime`] mutably (the RNG/counters can change
/// mid-evaluation) and the read [`Env`] plus the current `outer` row immutably.
pub(crate) struct EvalCtx<'x> {
    /// Connection runtime — RNG, change counters, bind parameters.
    pub rt: &'x mut Runtime,
    /// The shared read context (plan + catalog + pager) a subquery runs against: the
    /// subquery callbacks resolve `env.plan.subqueries[id]` and stream it through
    /// [`build_cursor`](crate::runner::build_cursor).
    pub env: Env<'x>,
    /// The current row, as the eval context's view of the outer a nested subquery
    /// would read. The subquery callbacks take the *same* row through their `regs`
    /// parameter (the portable path every [`EvalContext`] impl gets) and pass it to a
    /// correlated subplan, so this field mirrors `regs` rather than being read here;
    /// it is retained as part of the context shape the operators construct.
    #[allow(dead_code)]
    pub outer: &'x [Value],
}

impl FnContext for EvalCtx<'_> {
    fn now_unix_millis(&self) -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        // Real wall clock in the imperative shell. A pre-epoch clock (not physically
        // reachable here) degrades to 0 rather than panicking.
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_millis() as i64,
            Err(_) => 0,
        }
    }

    fn random_i64(&mut self) -> i64 {
        self.rt.random_i64()
    }

    fn fill_random(&mut self, buf: &mut [u8]) {
        self.rt.fill_random(buf)
    }

    fn last_insert_rowid(&self) -> i64 {
        self.rt.last_insert_rowid()
    }

    fn changes(&self) -> i64 {
        self.rt.changes()
    }

    fn total_changes(&self) -> i64 {
        self.rt.total_changes()
    }

    // The ephemeral JSON value-subtype channel (json1.html §3.4). The evaluator sets
    // the arg subtypes and reads back the result subtype around each function call;
    // the JSON functions read `arg_subtype` and set `set_result_subtype`. Backed by
    // the connection `Runtime` so the buffer is reused across calls and no `EvalCtx`
    // construction site changes (see `Runtime::arg_subtypes`).
    fn set_arg_subtypes(&mut self, s: &[u8]) {
        self.rt.set_arg_subtypes(s);
    }

    fn arg_subtype(&self, i: usize) -> u8 {
        self.rt.arg_subtype(i)
    }

    fn set_result_subtype(&mut self, st: u8) {
        self.rt.set_result_subtype(st);
    }

    fn take_result_subtype(&mut self) -> u8 {
        self.rt.take_result_subtype()
    }
}

impl EvalContext for EvalCtx<'_> {
    fn param(&self, index: usize) -> Result<Value> {
        self.rt.param(index)
    }

    /// Record a trigger body's `RAISE(IGNORE)` on the runtime (see
    /// [`Runtime::set_raise_ignore`](crate::runtime::Runtime::set_raise_ignore)). The
    /// evaluator returns a sentinel `Err` right after calling this; `fire_triggers`
    /// consumes the flag and converts it into a row-skip instead of surfacing the error.
    fn signal_raise_ignore(&mut self) {
        self.rt.set_raise_ignore();
    }

    // The subquery callbacks (five methods over three kinds — scalar, EXISTS, IN) resolve
    // `env.plan.subqueries[id]` (id guarded) and stream the subplan via `open_subquery`
    // (which applies the correlation rule). The evaluator applies `negated`/3VL wrapping
    // AROUND these, so each returns the RAW result (scalar value / bool / `Option<bool>`).
    //
    // CORRELATION and CACHING split here (lang_expr.html §12): a CORRELATED subplan depends
    // on the outer row, so it cannot share one result across all rows; an INELIGIBLE
    // correlated subplan therefore re-runs on every call, while an ELIGIBLE one (see
    // `correlated_memo_eligible`) is MEMOIZED by its correlation key — collapsing a
    // low-cardinality correlation from one subplan run per outer row to one per distinct key
    // (the O(n*n) -> O(n) win). An UNCORRELATED subplan yields the same result for every
    // outer row, so it is "evaluated only once and the result reused" — the first call runs
    // it and stores the result in the `Runtime`'s per-statement `subquery_cache` (keyed by
    // `id`), and every later call in the same statement reads that cached result instead of
    // re-running the subplan. That is correctness for a volatile subquery (a
    // `(SELECT random())` must repeat one value across the rows) as much as it removes the
    // per-outer-row re-scan for `x IN (SELECT ...)`.
    fn eval_scalar_subquery(&mut self, id: SubqueryId, regs: &[Value]) -> Result<Value> {
        let sub = self
            .env
            .plan
            .subqueries
            .get(id)
            .ok_or_else(|| Error::sql("scalar subquery id out of range"))?;
        // Uncorrelated: return the cached value if this subquery already ran once.
        if !sub.correlated && let Some(cached) = self.rt.cached_subquery(id) {
            return match cached {
                CachedSubquery::Scalar(v) => Ok(v.clone()),
                _ => Err(Error::sql("cached subquery kind mismatch for scalar subquery")),
            };
        }
        // Correlated + eligible: serve the memo on a HIT; on a miss remember the key so the
        // value computed below is stored under it (see the callbacks' doc for eligibility).
        let memo_key = correlated_memo_key(self.env, sub, regs, self.rt);
        if let Some(key) = &memo_key {
            if let Some(cached) = self.rt.cached_correlated_subquery(id, key) {
                return match cached {
                    CachedSubquery::Scalar(v) => Ok(v.clone()),
                    _ => Err(Error::sql(
                        "cached correlated subquery kind mismatch for scalar subquery",
                    )),
                };
            }
        }
        // First column of the first row; no rows is SQL NULL. Only the first row is
        // consumed — a scalar subquery ignores the rest. (`open_subquery` runs an
        // uncorrelated subplan with an EMPTY outer regardless of `regs`.)
        let value = {
            let mut cur = open_subquery(self.env, sub, regs)?;
            match cur.next_row(self.rt)? {
                Some(row) => first_col(row),
                None => Value::Null,
            }
        };
        if !sub.correlated {
            self.rt.cache_subquery(id, CachedSubquery::Scalar(value.clone()));
        } else if let Some(key) = memo_key {
            self.rt
                .cache_correlated_subquery(id, key, CachedSubquery::Scalar(value.clone()));
        }
        Ok(value)
    }

    // The row-value generalization of `eval_scalar_subquery`: it runs the SAME subplan,
    // takes the first row, and returns its `col`th column (NULL if there is no row). A
    // column-list UPDATE source `SET (a, b, …) = (SELECT …)` emits one
    // `ScalarSubqueryColumn` per target column, ALL sharing this `id`, so caching the
    // whole first ROW (not just column 0) is what lets those N reads share ONE subplan
    // run — the same once-per-statement reuse an uncorrelated scalar subquery gets
    // (lang_expr.html §12), extended across the N columns. A correlated source depends on
    // the outer (target) row, so it re-runs per call, exactly like the scalar path.
    fn eval_scalar_subquery_column(
        &mut self,
        id: SubqueryId,
        col: usize,
        regs: &[Value],
    ) -> Result<Value> {
        let sub = self
            .env
            .plan
            .subqueries
            .get(id)
            .ok_or_else(|| Error::sql("row-value subquery id out of range"))?;
        // Uncorrelated: serve the requested column from the cached first row.
        if !sub.correlated && let Some(cached) = self.rt.cached_subquery(id) {
            return match cached {
                CachedSubquery::FirstRow(row) => Ok(row.get(col).cloned().unwrap_or(Value::Null)),
                _ => Err(Error::sql("cached subquery kind mismatch for row-value subquery")),
            };
        }
        // Correlated + eligible: serve the requested column from the memoized first row on a
        // HIT; on a miss remember the key so the row materialized below is cached under it.
        // All N `ScalarSubqueryColumn` reads for one outer row share id+key, so they share
        // ONE subplan run (the same reuse the uncorrelated FirstRow cache gives).
        let memo_key = correlated_memo_key(self.env, sub, regs, self.rt);
        if let Some(key) = &memo_key {
            if let Some(cached) = self.rt.cached_correlated_subquery(id, key) {
                return match cached {
                    CachedSubquery::FirstRow(row) => {
                        Ok(row.get(col).cloned().unwrap_or(Value::Null))
                    }
                    _ => Err(Error::sql(
                        "cached correlated subquery kind mismatch for row-value subquery",
                    )),
                };
            }
        }
        // Materialize the first row (an EMPTY vec if the subplan produced none — the
        // cache's "no rows" sentinel). Only the first row is consumed, mirroring the
        // scalar path. (`open_subquery` runs an uncorrelated subplan with an EMPTY outer
        // regardless of `regs`; a correlated one reads `regs` as its outer.)
        let row: Row = {
            let mut cur = open_subquery(self.env, sub, regs)?;
            cur.next_row(self.rt)?.unwrap_or_default()
        };
        // The binder width-checks the subquery to exactly the name-list length and emits
        // only `col < width`, so `get(col)` is `Some` for a present row and `None` (=>
        // NULL) only when the subquery returned no row.
        let value = row.get(col).cloned().unwrap_or(Value::Null);
        if !sub.correlated {
            self.rt.cache_subquery(id, CachedSubquery::FirstRow(row));
        } else if let Some(key) = memo_key {
            self.rt
                .cache_correlated_subquery(id, key, CachedSubquery::FirstRow(row));
        }
        Ok(value)
    }

    fn eval_exists(&mut self, id: SubqueryId, regs: &[Value]) -> Result<bool> {
        let sub = self
            .env
            .plan
            .subqueries
            .get(id)
            .ok_or_else(|| Error::sql("EXISTS subquery id out of range"))?;
        // Uncorrelated: return the cached truth if this subquery already ran once.
        if !sub.correlated && let Some(cached) = self.rt.cached_subquery(id) {
            return match cached {
                CachedSubquery::Exists(b) => Ok(*b),
                _ => Err(Error::sql("cached subquery kind mismatch for EXISTS subquery")),
            };
        }
        // Correlated + eligible: serve the memoized truth on a HIT; on a miss remember the
        // key so the boolean computed below is cached under it.
        let memo_key = correlated_memo_key(self.env, sub, regs, self.rt);
        if let Some(key) = &memo_key {
            if let Some(cached) = self.rt.cached_correlated_subquery(id, key) {
                return match cached {
                    CachedSubquery::Exists(b) => Ok(*b),
                    _ => Err(Error::sql(
                        "cached correlated subquery kind mismatch for EXISTS subquery",
                    )),
                };
            }
        }
        // EXISTS is true iff the subplan yields at least one row; stop at the first
        // rather than draining.
        let exists = {
            let mut cur = open_subquery(self.env, sub, regs)?;
            cur.next_row(self.rt)?.is_some()
        };
        if !sub.correlated {
            self.rt.cache_subquery(id, CachedSubquery::Exists(exists));
        } else if let Some(key) = memo_key {
            self.rt
                .cache_correlated_subquery(id, key, CachedSubquery::Exists(exists));
        }
        Ok(exists)
    }

    fn eval_in_subquery(
        &mut self,
        id: SubqueryId,
        probe: &Value,
        meta: &CompareMeta,
        regs: &[Value],
    ) -> Result<Option<bool>> {
        let sub = self
            .env
            .plan
            .subqueries
            .get(id)
            .ok_or_else(|| Error::sql("IN subquery id out of range"))?;

        // UNCORRELATED: materialize the candidate set ONCE (cached by `id`), then probe
        // it with an O(1) hash lookup. The set holds the non-NULL candidate keys after
        // right-affinity coercion, folded by the collation; `set_has_null` / `any_rows`
        // carry the extra state SQL's three-valued `IN` needs. `probe_in_set`
        // reproduces the linear `compare_for_eq` 3VL of the correlated path below
        // EXACTLY (see tests/subquery_cache.rs), so caching changes cost, not results.
        if !sub.correlated {
            if let Some(cached) = self.rt.cached_subquery(id) {
                return match cached {
                    CachedSubquery::InSet { set, set_has_null, any_rows } => {
                        Ok(probe_in_set(set, *set_has_null, *any_rows, probe, meta))
                    }
                    _ => Err(Error::sql("cached subquery kind mismatch for IN subquery")),
                };
            }
            // Cache miss: drain the subplan into the candidate set exactly once.
            let (set, set_has_null, any_rows) =
                materialize_in_set(self.env, sub, regs, meta, self.rt)?;
            // Probe on the freshly-built set, THEN cache it (the borrow of `self.rt`
            // that drove the subplan has ended, so storing into `self.rt` is free).
            let result = probe_in_set(&set, set_has_null, any_rows, probe, meta);
            self.rt.cache_subquery(id, CachedSubquery::InSet { set, set_has_null, any_rows });
            return Ok(result);
        }

        // CORRELATED + eligible: memoize the candidate SET keyed by the correlation key.
        // The set depends only on the correlated inputs, so two outer rows sharing a key
        // reuse ONE materialization; the probe is THIS row's subject, applied fresh on every
        // hit, so a varying probe over a shared set stays correct. Materialization is the
        // same `materialize_in_set` the uncorrelated branch uses, and `probe_in_set`
        // reproduces the streaming 3VL exactly, so the memo changes cost, not the answer.
        if let Some(key) = correlated_memo_key(self.env, sub, regs, self.rt) {
            if let Some(cached) = self.rt.cached_correlated_subquery(id, &key) {
                return match cached {
                    CachedSubquery::InSet { set, set_has_null, any_rows } => {
                        Ok(probe_in_set(set, *set_has_null, *any_rows, probe, meta))
                    }
                    _ => Err(Error::sql(
                        "cached correlated subquery kind mismatch for IN subquery",
                    )),
                };
            }
            let (set, set_has_null, any_rows) =
                materialize_in_set(self.env, sub, regs, meta, self.rt)?;
            let result = probe_in_set(&set, set_has_null, any_rows, probe, meta);
            self.rt.cache_correlated_subquery(
                id,
                key,
                CachedSubquery::InSet { set, set_has_null, any_rows },
            );
            return Ok(result);
        }

        // CORRELATED (ineligible): re-run the subplan for this outer row, streaming the
        // candidates through the linear 3VL comparison. The pre-memo fallback — a correlated
        // candidate set depends on the outer row and cannot be reused when memo is off.
        let mut cur = open_subquery(self.env, sub, regs)?;

        // Apply the probe's affinity once, mirroring the evaluator's static IN
        // (expr/eval.rs `InList`).
        let p = coerce(probe.clone(), meta.apply_left);
        let mut any_rows = false;
        let mut saw_null = false;
        while let Some(row) = cur.next_row(self.rt)? {
            any_rows = true;
            let c = coerce(first_col(row), meta.apply_right);
            match compare_for_eq(&p, &c, meta.collation) {
                // An equal candidate settles it TRUE regardless of any NULLs seen.
                Some(Ordering::Equal) => return Ok(Some(true)),
                Some(_) => {}
                // A NULL on either side makes this one comparison unknown.
                None => saw_null = true,
            }
        }
        // 3VL after scanning every candidate: an EMPTY set is FALSE even for a NULL
        // probe (`x IN ()` is FALSE); otherwise no equal match with a NULL involved is
        // unknown; else FALSE. `negated` is applied by the evaluator, not here.
        Ok(if !any_rows {
            Some(false)
        } else if saw_null {
            None
        } else {
            Some(false)
        })
    }

    fn eval_in_subquery_row(
        &mut self,
        id: SubqueryId,
        probe: &[Value],
        metas: &[CompareMeta],
        regs: &[Value],
    ) -> Result<Option<bool>> {
        let sub = self
            .env
            .plan
            .subqueries
            .get(id)
            .ok_or_else(|| Error::sql("row-value IN subquery id out of range"))?;

        // UNCORRELATED: materialize the candidate ROWS once (cached by `id`), then probe
        // them with the tuple 3VL. A tuple cannot use the scalar path's `HashSet` — a
        // per-element NULL makes membership three-valued at the row level (see
        // [`row_match3`]) — so the candidates are kept as rows and scanned. Caching still
        // runs the subplan exactly once (lang_expr.html §12), matching the scalar path's
        // structure: it changes cost, not the answer.
        if !sub.correlated {
            if let Some(cached) = self.rt.cached_subquery(id) {
                return match cached {
                    CachedSubquery::InRows(rows) => Ok(probe_in_rows(rows, probe, metas)),
                    _ => Err(Error::sql("cached subquery kind mismatch for row-value IN subquery")),
                };
            }
            // Cache miss: drain the subplan into the candidate rows exactly once.
            let rows: Vec<Row> = materialize_in_rows(self.env, sub, regs, self.rt)?;
            let result = probe_in_rows(&rows, probe, metas);
            self.rt.cache_subquery(id, CachedSubquery::InRows(rows));
            return Ok(result);
        }

        // CORRELATED + eligible: memoize the candidate ROWS keyed by the correlation key,
        // then probe THIS row's tuple against them (same reasoning as the scalar `IN` memo).
        // Materialization is the same `materialize_in_rows` the uncorrelated branch uses, and
        // `probe_in_rows` reproduces the streaming tuple 3VL exactly.
        if let Some(key) = correlated_memo_key(self.env, sub, regs, self.rt) {
            if let Some(cached) = self.rt.cached_correlated_subquery(id, &key) {
                return match cached {
                    CachedSubquery::InRows(rows) => Ok(probe_in_rows(rows, probe, metas)),
                    _ => Err(Error::sql(
                        "cached correlated subquery kind mismatch for row-value IN subquery",
                    )),
                };
            }
            let rows = materialize_in_rows(self.env, sub, regs, self.rt)?;
            let result = probe_in_rows(&rows, probe, metas);
            self.rt
                .cache_correlated_subquery(id, key, CachedSubquery::InRows(rows));
            return Ok(result);
        }

        // CORRELATED (ineligible): re-run the subplan for this outer row, streaming each
        // candidate row through the tuple AND3 without materializing. The pre-memo fallback;
        // a correlated candidate set depends on the outer row and cannot be reused when memo
        // is off. Same 3VL folding as [`probe_in_rows`].
        let mut cur = open_subquery(self.env, sub, regs)?;
        let mut saw_unknown = false;
        while let Some(row) = cur.next_row(self.rt)? {
            match row_match3(probe, &row, metas) {
                // A fully-equal candidate row settles it TRUE regardless of any unknowns.
                Some(true) => return Ok(Some(true)),
                Some(false) => {}
                // This row is an unknown (NULL-blocked) match — remember it in case no
                // row is a definite match.
                None => saw_unknown = true,
            }
        }
        // No full match: an unknown row makes the whole result UNKNOWN; otherwise FALSE
        // (an EMPTY subquery never sets `saw_unknown`, so it is FALSE — `NOT IN` empty is
        // TRUE). `negated` is applied by the evaluator, not here.
        Ok(if saw_unknown { None } else { Some(false) })
    }
}

/// Whether the correlated-subquery memo may serve `sub` in this statement — the three
/// mandatory eligibility guards, in ONE predicate so all five callbacks share exactly the
/// same rule and cannot drift:
///
/// * (a) READ-ONLY statement (`!plan.mutates`) — the DML-staleness gate. A memo keyed by
///   `(id, outer-values)` assumes the subquery result is STABLE for a key across the whole
///   statement. That holds for a read (a consistent snapshot), but NOT when the enclosing
///   statement writes a table the subquery reads: `UPDATE t SET c=(SELECT count(*) FROM t x
///   WHERE x.g=t.g)` legitimately yields DIFFERENT counts for two same-`g` rows as `t`
///   mutates row by row, so a per-key memo would serve a stale answer. `deterministic` does
///   not capture this; `!mutates` is the directive's simple, safe rule.
/// * (b) DETERMINISTIC subplan (`sub.deterministic`) — a volatile subplan (`random()`,
///   `CURRENT_*`, or an un-analyzed materialized/CTE body) must re-draw per outer row, so it
///   is never memoized (a memo would wrongly repeat one draw across a key).
/// * (c) NON-EMPTY correlation key (`correlated && !correlated_cols.is_empty()`) — an empty
///   key would collapse EVERY outer row to a single entry, serving unrelated rows one
///   result. (An uncorrelated subplan has its own `id`-keyed cache and never reaches here.)
///
/// All three must hold; otherwise the correlated subplan re-runs per outer row — the exact
/// pre-memo behavior, the safe fallback. Correctness is paramount: when in doubt, re-run.
fn correlated_memo_eligible(plan: &Plan, sub: &SubPlan) -> bool {
    !plan.mutates && sub.deterministic && sub.correlated && !sub.correlated_cols.is_empty()
}

/// The correlation cache key for `sub` at outer row `regs`, or `None` when the correlated
/// memo must NOT apply — `sub` is uncorrelated/ineligible ([`correlated_memo_eligible`]) or
/// the memo is disabled (the test toggle). `Some(key)` means "eligible: look up / store
/// under this key". The key is OWNED (`corr_key` copies the referenced outer cells), so it
/// outlives the shared `env`/`sub` borrow and can be handed to the `&mut` cache store.
fn correlated_memo_key(
    env: Env,
    sub: &SubPlan,
    regs: &[Value],
    rt: &Runtime,
) -> Option<Vec<CorrCell>> {
    if rt.correlated_cache_disabled() || !correlated_memo_eligible(env.plan, sub) {
        return None;
    }
    // Planner invariant (the binder's correlation analysis): every correlated col is
    // `< outer_width <= regs.len()`, and the callbacks hand the outer row as `regs`. Assert
    // it under test so a planner bug is loud at its cause, not a later out-of-bounds inside
    // `corr_key`.
    debug_assert!(
        sub.correlated_cols.iter().all(|&c| c < regs.len()),
        "correlated_cols out of range for the outer row (planner invariant violated)"
    );
    Some(corr_key(regs, &sub.correlated_cols))
}

/// Drain an `IN (subquery)` subplan into the candidate SET that both the uncorrelated cache
/// and the correlated memo probe — the ONE materialization, so a memo entry is byte-identical
/// to what the streaming path would compute. Each candidate takes the right-affinity coercion
/// and collation fold SQL's `IN` requires; a NULL candidate sets `set_has_null` rather than
/// inserting a never-matched key. Returns `(set, set_has_null, any_rows)` for [`probe_in_set`].
fn materialize_in_set<'e>(
    env: Env<'e>,
    sub: &'e SubPlan,
    regs: &'e [Value],
    meta: &CompareMeta,
    rt: &mut Runtime,
) -> Result<(HashSet<CellKey>, bool, bool)> {
    let mut cur = open_subquery(env, sub, regs)?;
    let mut set: HashSet<CellKey> = HashSet::new();
    let mut set_has_null = false;
    let mut any_rows = false;
    while let Some(row) = cur.next_row(rt)? {
        any_rows = true;
        let c = coerce(first_col(row), meta.apply_right);
        // A NULL candidate never matches, but its presence makes a non-matching probe
        // UNKNOWN — record the flag instead of inserting a meaningless key.
        if c.is_null() {
            set_has_null = true;
        } else {
            set.insert(cell_key(&c, meta.collation));
        }
    }
    Ok((set, set_has_null, any_rows))
}

/// Drain a row-value `IN (subquery)` subplan into the candidate ROWS that both the
/// uncorrelated cache and the correlated memo probe. Unlike the scalar set, tuple membership
/// is scanned (a per-element NULL makes a row an UNKNOWN match — see [`row_match3`]), so the
/// rows are kept verbatim for [`probe_in_rows`]. The ONE materialization, byte-identical to
/// the streaming path.
fn materialize_in_rows<'e>(
    env: Env<'e>,
    sub: &'e SubPlan,
    regs: &'e [Value],
    rt: &mut Runtime,
) -> Result<Vec<Row>> {
    let mut cur = open_subquery(env, sub, regs)?;
    let mut rows: Vec<Row> = Vec::new();
    while let Some(row) = cur.next_row(rt)? {
        rows.push(row);
    }
    Ok(rows)
}

/// Probe an already-materialized `IN` candidate set for `probe`, reproducing the
/// linear-scan three-valued logic of the streaming path with an O(1) hash lookup.
///
/// Parameters mirror [`CachedSubquery::InSet`]: `set` holds the non-NULL candidate
/// keys (post-right-affinity, collation-folded), `set_has_null` whether any candidate
/// coerced to NULL, `any_rows` whether the subplan yielded any row. The probe's left
/// affinity is applied here (once), matching the evaluator's static-IN path. Returns
/// the RAW `Option<bool>` (the evaluator applies `negated` around it).
///
/// This is EQUIVALENT to the streaming loop because of a key-equality invariant the
/// engine already relies on (it underpins the hash join too): for non-NULL values
/// `a`, `b` — after the same affinity and under collation `c` —
/// `cell_key(a, c) == cell_key(b, c)` iff `compare_for_eq(a, b, c) == Some(Equal)`.
/// So "the probe's key is in the set" is exactly "some candidate compared Equal".
/// The three branches then map to the streaming 3VL: a NULL probe is UNKNOWN against a
/// non-empty set but FALSE against the empty set (`x IN ()` is FALSE); a hit is TRUE;
/// otherwise a NULL candidate makes it UNKNOWN; else FALSE.
fn probe_in_set(
    set: &HashSet<CellKey>,
    set_has_null: bool,
    any_rows: bool,
    probe: &Value,
    meta: &CompareMeta,
) -> Option<bool> {
    // Coerce by the probe's left affinity, but CLONE only when there is affinity to
    // apply: with no left affinity (the common case) the probe is borrowed straight
    // through, so a text/blob probe is not re-allocated on every outer row. This is the
    // hot path — one probe per outer row on a cache HIT, with no subplan re-run to dwarf
    // the clone (unlike the correlated loop above, which re-runs the whole subplan).
    let coerced;
    let p: &Value = match meta.apply_left {
        Some(a) => {
            coerced = apply_affinity(probe.clone(), a);
            &coerced
        }
        None => probe,
    };
    if p.is_null() {
        // A NULL probe is UNKNOWN against any non-empty set, but `x IN ()` is FALSE
        // even for NULL — so the empty set is the one case that is not UNKNOWN.
        return if any_rows { None } else { Some(false) };
    }
    if set.contains(&cell_key(p, meta.collation)) {
        Some(true)
    } else if set_has_null {
        None
    } else {
        Some(false)
    }
}

/// Probe already-materialized candidate ROWS for the tuple `probe`, reproducing the
/// streaming tuple three-valued logic (rowvalue.html §2.2). Returns the RAW
/// `Option<bool>` (the evaluator applies `negated`): `Some(true)` if any row equals the
/// probe element-wise; else `None` (UNKNOWN) if some row was an unknown match — every
/// non-NULL element equal, blocked only by a NULL; else `Some(false)`. An EMPTY `rows`
/// yields `Some(false)`, so `(…) NOT IN (empty)` is TRUE. This is the tuple counterpart
/// of [`probe_in_set`]; unlike the scalar set, tuple membership must scan rows because a
/// per-element NULL makes the row-level match three-valued rather than a hash lookup.
fn probe_in_rows(rows: &[Row], probe: &[Value], metas: &[CompareMeta]) -> Option<bool> {
    let mut saw_unknown = false;
    for row in rows {
        match row_match3(probe, row, metas) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => saw_unknown = true,
        }
    }
    if saw_unknown {
        None
    } else {
        Some(false)
    }
}

/// Three-valued equality of the `probe` tuple against ONE `candidate` row (rowvalue.html
/// §2.2): the AND3 over the per-element comparisons. `Some(true)` iff EVERY element
/// compares equal; `Some(false)` as soon as one element is definitely unequal (AND3
/// short-circuits on a FALSE regardless of any earlier unknown); else `None` — every
/// non-NULL element equal with at least one element NULL (on the probe or the candidate).
fn row_match3(probe: &[Value], candidate: &Row, metas: &[CompareMeta]) -> Option<bool> {
    // Debug-only invariant checks: the binder pins BOTH the subjects (hence `probe`) and
    // the width-checked subquery (hence `candidate`) to exactly `metas.len()` columns, so
    // these catch an engine bug that produced a mismatched width UNDER TEST. Release stays
    // total — the loop's `.get()` fallback degrades a too-narrow row to UNKNOWN rather than
    // panicking — so the asserts add safety in debug without changing release behavior.
    debug_assert_eq!(probe.len(), metas.len(), "probe width must equal the per-element metadata");
    debug_assert_eq!(candidate.len(), metas.len(), "candidate row width must equal the per-element metadata");
    let mut result = Some(true);
    for (i, meta) in metas.iter().enumerate() {
        // Both sides use `.get()` so the posture is symmetric and no index can panic; a
        // narrower row (the engine-invariant violation the debug asserts above catch under
        // test) reads the missing column as an unknown rather than panicking, keeping the
        // scan total.
        let elem = match (probe.get(i), candidate.get(i)) {
            (Some(p), Some(cand)) => eq3_elem(p, cand, meta),
            _ => None,
        };
        match elem {
            Some(Ordering::Equal) => {}
            Some(_) => return Some(false),
            None => result = None,
        }
    }
    result
}

/// Compare one tuple element `probe[i]` against a candidate column under its own
/// [`CompareMeta`] (affinity + collation), returning the `compare_for_eq` ordering. The
/// affinity is applied per side, but only CLONING when there is affinity to apply (the
/// common no-affinity case borrows straight through, so an integer/text column is not
/// re-allocated per candidate row) — the same borrow-when-possible idiom as
/// [`probe_in_set`]. A NULL on either side yields `None` (unknown) from `compare_for_eq`.
fn eq3_elem(probe: &Value, candidate: &Value, meta: &CompareMeta) -> Option<Ordering> {
    let coerced_p;
    let p: &Value = match meta.apply_left {
        Some(a) => {
            coerced_p = apply_affinity(probe.clone(), a);
            &coerced_p
        }
        None => probe,
    };
    let coerced_c;
    let c: &Value = match meta.apply_right {
        Some(a) => {
            coerced_c = apply_affinity(candidate.clone(), a);
            &coerced_c
        }
        None => candidate,
    };
    compare_for_eq(p, c, meta.collation)
}

/// Open a subquery's plan as a streaming cursor against the correct outer row — the
/// load-bearing CORRELATION RULE, in one place for all the subquery callbacks. A correlated
/// subplan's leaves prepend the outer row and its `Column(i)` for `i < outer_width`
/// reads it, so it runs with `regs` (the current outer row); a non-correlated subplan
/// was bound with `outer_width` 0, so it MUST run with an EMPTY outer — passing `regs`
/// would shift its columns and corrupt results. Takes [`Env`] by value (it is `Copy`),
/// not `&self`, so the returned cursor borrows the plan/outer rather than the whole
/// `EvalCtx`, leaving `self.rt` free to be reborrowed while the cursor is driven.
///
/// This always runs the subplan afresh. The once-per-statement reuse required for an
/// uncorrelated subquery (lang_expr.html §12) is handled by the subquery callbacks above,
/// which consult / populate the `Runtime`'s `subquery_cache` around this call; a
/// correlated subplan legitimately re-opens here on every outer row.
fn open_subquery<'e>(
    env: Env<'e>,
    sub: &'e SubPlan,
    regs: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    let outer: &[Value] = if sub.correlated { regs } else { &[] };
    build_cursor(&sub.plan, env, outer)
}

/// The first column of a subquery result row, or `Value::Null` for an unexpectedly
/// empty row. A subquery bound as scalar / `IN` produces exactly one output column, so
/// this is that value; the NULL fallback keeps the read total rather than panicking.
fn first_col(row: Row) -> Value {
    row.into_iter().next().unwrap_or(Value::Null)
}

/// Coerce `v` by an optional comparison affinity: apply it when present, pass the value
/// through unchanged when absent. The single home for the `IN` probe/candidate affinity
/// idiom (mirrors `minisqlite-expr`'s private `apply_opt`), used at every candidate site
/// and the correlated probe. Owned in / owned out, because those sites already hold an
/// owned [`Value`] (`first_col(row)` / `probe.clone()`); the hot cache-hit probe in
/// [`probe_in_set`] deliberately does NOT use this — it borrows to avoid a per-row clone.
fn coerce(v: Value, aff: Option<Affinity>) -> Value {
    match aff {
        Some(a) => apply_affinity(v, a),
        None => v,
    }
}
