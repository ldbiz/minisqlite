//! [`PlanCtx`] — the mutable per-statement context threaded through binding and
//! compilation. It owns the two side tables that end up in the final [`Plan`]
//! (`subqueries` and `ctes`), borrows the function registry and catalog the binder
//! resolves against, and tracks bind-parameter numbering across the whole
//! statement.
//!
//! One `PlanCtx` is created per top-level statement (in the [`QueryPlanner`]) and
//! passed by `&mut` to every binder/compiler function, so parameter numbers and
//! subquery ids are assigned in one shared, monotonic space.
//!
//! [`Plan`]: crate::Plan
//! [`QueryPlanner`]: crate::QueryPlanner

use minisqlite_catalog::Catalog;
use minisqlite_functions::FunctionRegistry;
use minisqlite_sql::BindParam;
use minisqlite_types::{Error, Result};

use crate::plan::{CtePlan, SubPlan};

/// Per-statement compilation state. Lifetime `'a` borrows the registry and catalog
/// (both read-only for the duration of one `plan()` call).
///
/// INVARIANT for maintainers: every mutable field a *trial* bind can advance
/// (`subqueries`, `ctes`, `next_max`, `named`) is snapshotted by [`Savepoint`] and
/// rolled back by [`restore`](Self::restore). If you add another such field, extend BOTH
/// or a correlated-subquery re-bind will silently carry the discarded trial's mutation
/// forward (a wrong subquery id / parameter number surfacing in an unrelated statement).
pub struct PlanCtx<'a> {
    /// The built-in function registry the binder resolves scalar/aggregate names
    /// against (owned by the [`QueryPlanner`](crate::QueryPlanner)).
    pub registry: &'a FunctionRegistry,
    /// The schema store, borrowed to look up tables during FROM compilation.
    pub catalog: &'a dyn Catalog,
    /// Expression subqueries collected during binding; moved into `Plan::subqueries`.
    /// An `EvalExpr` subquery id is an index into this vector.
    pub subqueries: Vec<SubPlan>,
    /// Materialized / recursive CTEs and inline derived tables; moved into `Plan::ctes`.
    /// A reference lowers to a `CteScan { id }` where `id` indexes this vector. Populated
    /// by `compile/cte.rs` (the `WITH` clause) and `compile/from.rs` (inline `FROM (…)`).
    pub ctes: Vec<CtePlan>,
    /// The largest bind-parameter number assigned so far. `?` and a fresh named
    /// parameter each take `next_max + 1`; `?N` bumps it to at least `N`.
    next_max: u32,
    /// Named parameters already assigned, in first-seen order. Reusing the exact
    /// same name (including its sigil) reuses the number, per SQLite's rule.
    named: Vec<(String, u32)>,
}

/// A rollback point produced by [`PlanCtx::savepoint`] and consumed by
/// [`PlanCtx::restore`]. Opaque snapshot of the append-only side-table lengths and the
/// parameter-numbering state; carries no borrow, so it can outlive the `&PlanCtx` it
/// was taken from.
#[derive(Clone, Copy)]
pub struct Savepoint {
    subqueries: usize,
    ctes: usize,
    next_max: u32,
    named: usize,
}

impl<'a> PlanCtx<'a> {
    /// A fresh context for one statement.
    pub fn new(registry: &'a FunctionRegistry, catalog: &'a dyn Catalog) -> Self {
        PlanCtx { registry, catalog, subqueries: Vec::new(), ctes: Vec::new(), next_max: 0, named: Vec::new() }
    }

    /// Register a compiled subquery and return its [`SubqueryId`](minisqlite_expr::SubqueryId)
    /// (its index in `subqueries`).
    pub fn register_subquery(&mut self, sub: SubPlan) -> usize {
        let id = self.subqueries.len();
        self.subqueries.push(sub);
        id
    }

    /// Capture a rollback point for a *trial* bind. The two side tables (`subqueries`,
    /// `ctes`) are append-only and the parameter state (`next_max`, `named`) is
    /// monotonic within a compile, so a snapshot of their lengths / value is enough to
    /// undo everything a speculative compilation added.
    ///
    /// Used by the correlated-subquery compiler: it trial-binds a subquery to discover
    /// whether it is correlated, and — when it must re-bind with a different register
    /// base — [`restore`](Self::restore)s to this point first, so the re-bind reproduces
    /// the *identical* subquery ids and `?`/named parameter numbers the trial assigned
    /// (parameter numbering across the whole statement must be unaffected by the retry).
    pub fn savepoint(&self) -> Savepoint {
        Savepoint {
            subqueries: self.subqueries.len(),
            ctes: self.ctes.len(),
            next_max: self.next_max,
            named: self.named.len(),
        }
    }

    /// Roll back to a [`savepoint`](Self::savepoint): drop any subqueries/CTEs the trial
    /// registered and reset parameter numbering to exactly what it was. Truncation is
    /// sound because both side tables are append-only and `named` is only ever pushed to
    /// (a reused name never appends), so the retained prefix is precisely the pre-trial
    /// state.
    pub fn restore(&mut self, sp: Savepoint) {
        self.subqueries.truncate(sp.subqueries);
        self.ctes.truncate(sp.ctes);
        self.next_max = sp.next_max;
        self.named.truncate(sp.named);
    }

    /// Assign (or look up) the 1-based number for a bind parameter, following
    /// SQLite's rules (`lang_expr.html` §"Parameters"):
    /// * `?` — one greater than the largest number assigned so far.
    /// * `?N` — the literal `N` (which also raises the running maximum).
    /// * `:name` / `@name` / `$name` — the number already given to that exact
    ///   spelling, else a freshly allocated next number.
    ///
    /// The returned value is the 1-based index the executor's
    /// `EvalContext::param(index)` expects, wrapped in `EvalExpr::Param`.
    pub fn param_index(&mut self, p: &BindParam) -> Result<usize> {
        let n = match p {
            BindParam::Anonymous => self.bump_max()?,
            BindParam::Numbered(n) => {
                if *n == 0 {
                    return Err(Error::sql("bind parameter ?0 is not allowed (parameters are 1-based)"));
                }
                if *n > self.next_max {
                    self.next_max = *n;
                }
                *n
            }
            BindParam::Named(name) => {
                if let Some((_, num)) = self.named.iter().find(|(existing, _)| existing == name) {
                    *num
                } else {
                    let num = self.bump_max()?;
                    self.named.push((name.clone(), num));
                    num
                }
            }
        };
        Ok(n as usize)
    }

    /// Allocate the next auto-numbered parameter (`?` or a fresh named parameter),
    /// bumping the running maximum. Errors rather than overflowing past `u32::MAX`
    /// on the absurd boundary (e.g. `?4294967295` then `?`), matching SQLite's
    /// refusal of too many variables instead of a debug panic / silent wrap.
    fn bump_max(&mut self) -> Result<u32> {
        self.next_max =
            self.next_max.checked_add(1).ok_or_else(|| Error::sql("too many SQL variables"))?;
        Ok(self.next_max)
    }
}
