//! `WITH` (common table expressions): compile a SELECT's leading `WITH` clause,
//! materializing each CTE into [`Plan::ctes`](crate::plan::Plan::ctes) and compiling
//! the body against them. Handles ordinary CTEs and `WITH RECURSIVE` (spec
//! `spec/sqlite-doc/lang_with.html`).
//!
//! Delegation seam: [`compile_with`] is the single entrypoint `compile_select_scoped`
//! routes a `WITH`-bearing SELECT through, so the CTE feature lands here rather than in
//! the SELECT orchestrator.
//!
//! # How a CTE reference resolves (the CTE scope stack)
//!
//! A `FROM cte_name` is resolved by the FROM compiler ([`crate::compile::from`]) in two
//! phases: Phase 1 ([`resolve_table`](crate::compile::from)) builds the visible
//! [`Source`](crate::bind::Source), and Phase 2 ([`build_source_leaf`](crate::compile::from))
//! builds the access leaf. Both must agree on the CTE's synthetic schema and its
//! pre-registered [`CtePlan`] id.
//!
//! Phase 1's `resolve_from(catalog, from, base_offset)` is a fixed seam owned by
//! `compile/select.rs`: it takes only the catalog, not the [`PlanCtx`], so a CTE
//! registry cannot be threaded through its parameters without changing that owned seam.
//! We therefore expose the registry as a **thread-local scope stack** ([`CTE_SCOPE`])
//! that Phase 1 and Phase 2 consult via [`lookup_cte`]. It is a stack because CTE
//! visibility nests exactly like lexical scope: the CTEs of a `WITH` are visible to its
//! body and to every subquery inside it, an inner `WITH` (in a subquery) shadows/extends
//! the outer set, and a later CTE in one `WITH` sees the earlier ones. [`compile_with`]
//! pushes this `WITH`'s entries, compiles the body, and a [`CteScopeGuard`] pops them on
//! return (including an early `?`), so the stack stays balanced across the
//! correlated-subquery trial/rebind and never leaks a CTE into a sibling statement.
//!
//! (Planning is single-threaded per statement, so the thread-local is a per-statement
//! ambient value, never shared across concurrent planners.)

use std::cell::{Cell, RefCell};

use minisqlite_sql::{
    CompoundOp, Cte, JoinTree, Select, SelectBody, SelectCore, TableOrSubquery, With,
};
use minisqlite_types::{Error, Result};

use crate::bind::scope::SynthCol;
use crate::bind::Scope;
use crate::compile::from::derived_schema;
use crate::compile::select::{build_limit, compile_body, compile_select};
use crate::plan::{CtePlan, PlanNode};
use crate::plan_ctx::PlanCtx;

// ---------------------------------------------------------------------------
// The CTE scope stack (see the module docs for why this is a thread-local).
// ---------------------------------------------------------------------------

/// How a reference to a CTE lowers to an access leaf.
#[derive(Clone, Copy)]
pub(crate) enum CteLeaf {
    /// A materialized / recursive CTE at `Plan::ctes[id]`: a reference is a
    /// [`PlanNode::CteScan`]`{ id }`.
    Scan(usize),
    /// The recursive CTE currently being compiled, referenced from its OWN recursive
    /// step: a reference is a [`PlanNode::RecursiveScan`] reading the working table (the
    /// previous iteration's rows), never a `CteScan` (which would re-enter the fixpoint).
    Recursive,
    /// A Phase-1, schema-only registration (via [`push_cte_schemas`]) used by
    /// [`derived_schema`] to resolve a WITH-bearing FROM subquery's own CTE names when the
    /// enclosing query computes the subquery's schema — Phase 1 runs before the Phase-2
    /// `compile_with` that registers the real [`CtePlan`]s. Only the entry's columns are
    /// ever read (by `resolve_table`); the leaf carries no id because none is assigned yet.
    /// It is popped before Phase 2, so it must never reach the leaf builder — the Phase-2
    /// [`build_source_leaf`](crate::compile::from) treats it as an internal error.
    SchemaOnly,
}

/// What [`lookup_cte`] returns to the FROM compiler: the synthetic columns Phase 1 needs
/// to build the [`Source`](crate::bind::Source), and the leaf kind Phase 2 needs.
pub(crate) struct CteLookup {
    pub(crate) columns: Vec<SynthCol>,
    pub(crate) leaf: CteLeaf,
}

/// One CTE visible in the current compilation scope.
struct CteEntry {
    name: String,
    columns: Vec<SynthCol>,
    leaf: CteLeaf,
}

thread_local! {
    /// The stack of CTEs in scope, innermost (most recently pushed) last. Empty except
    /// while a `WITH`-bearing SELECT (or one enclosing it) is being compiled.
    static CTE_SCOPE: RefCell<Vec<CteEntry>> = const { RefCell::new(Vec::new()) };
}

/// Resolve a CTE by name (ASCII case-insensitive), searching innermost-scope-first so an
/// inner `WITH` shadows an outer CTE of the same name. Returns `None` when the name is
/// not a visible CTE (the FROM compiler then falls back to the catalog).
pub(crate) fn lookup_cte(name: &str) -> Option<CteLookup> {
    CTE_SCOPE.with(|scope| {
        scope
            .borrow()
            .iter()
            .rev()
            .find(|e| e.name.eq_ignore_ascii_case(name))
            .map(|e| CteLookup { columns: e.columns.clone(), leaf: e.leaf })
    })
}

fn cte_scope_len() -> usize {
    CTE_SCOPE.with(|scope| scope.borrow().len())
}

fn push_cte_entry(entry: CteEntry) {
    CTE_SCOPE.with(|scope| scope.borrow_mut().push(entry));
}

fn truncate_cte_scope(len: usize) {
    CTE_SCOPE.with(|scope| scope.borrow_mut().truncate(len));
}

/// Restores the CTE scope stack to its length at construction, on drop. Constructed at
/// the top of [`compile_with`] (and by [`push_cte_schemas`]) so an early `?` return from
/// body/schema compilation cannot leak a `WITH`'s CTEs into an enclosing or sibling scope.
pub(crate) struct CteScopeGuard {
    restore_len: usize,
}

impl CteScopeGuard {
    fn new() -> Self {
        CteScopeGuard { restore_len: cte_scope_len() }
    }
}

impl Drop for CteScopeGuard {
    fn drop(&mut self) {
        truncate_cte_scope(self.restore_len);
    }
}

/// Sets the CTE scope stack ASIDE (leaving it empty) for the guard's lifetime, restoring
/// it on drop. Used only by the view expander ([`crate::compile::view`]).
///
/// A view is a self-contained schema object: its body's FROM must resolve only the view's
/// OWN `WITH` CTEs and base tables/views — never a CTE from the query that *referenced*
/// the view, which is still on this shared stack while the view is expanded. Without this,
/// an outer `WITH x AS (…) SELECT … FROM a_view` where the view's body has an unrelated
/// `FROM x` would silently bind that `x` to the outer CTE instead of erroring — a wrong
/// result, not a loud failure. Holding this across both the view's schema computation
/// (Phase 1) and its body compilation (Phase 2) matches how real SQLite compiles a view
/// against a fresh context. The view's own `WITH` still works: it pushes onto and pops
/// from the now-empty stack via the normal [`CteScopeGuard`], nested strictly inside this
/// isolation.
pub(crate) struct IsolatedCteScope {
    saved: Vec<CteEntry>,
}

impl IsolatedCteScope {
    /// Take the current CTE scope aside, leaving an empty stack until the guard drops.
    pub(crate) fn enter() -> Self {
        let saved = CTE_SCOPE.with(|scope| std::mem::take(&mut *scope.borrow_mut()));
        IsolatedCteScope { saved }
    }
}

impl Drop for IsolatedCteScope {
    fn drop(&mut self) {
        // Restore exactly what we set aside, discarding whatever is on the stack now. This
        // is a WHOLESALE replace, not a truncate, so it rests on a load-bearing invariant:
        // any [`CteScopeGuard`] the view's OWN `WITH` pushed must be STRICTLY NESTED inside
        // this guard — created AND dropped within the `compile_view`/`view_schema` call that
        // holds this isolation — so the stack is empty again by the time we drop. It holds
        // today because `compile_select` creates+drops that guard within its own call. If a
        // future refactor let a `CteScopeGuard` outlive this isolation, the `mem::take` here
        // would silently discard those still-live `WITH` entries; keep the nesting strict.
        CTE_SCOPE.with(|scope| {
            let mut stack = scope.borrow_mut();
            // The strict-nesting invariant above means every entry pushed during this
            // isolation was already popped by its own (inner, earlier-dropped) guard, so the
            // stack is empty here. Assert it so a refactor that broke the nesting fails LOUD
            // instead of silently discarding live `WITH` entries via the take below.
            debug_assert!(
                stack.is_empty(),
                "IsolatedCteScope::drop expects a balanced (empty) stack; a CteScopeGuard \
                 outlived the isolation, so its WITH entries would be silently discarded",
            );
            *stack = std::mem::take(&mut self.saved);
        });
    }
}

/// Register the *schemas* of a `WITH` clause's CTEs into the scope for the lifetime of the
/// returned guard, WITHOUT compiling their bodies (Phase 1, catalog-only).
///
/// Used by [`derived_schema`] so a WITH-bearing FROM subquery
/// (`FROM (WITH c AS … SELECT … FROM c)`) can resolve its own CTE names when the enclosing
/// query computes the subquery's schema. Phase 1 runs before the Phase-2 `compile_with`
/// (reached via `compile_derived` → `compile_select`) that registers the real
/// [`CtePlan`]s, so at Phase 1 the inner CTEs are not yet visible; this closes that gap.
///
/// Each entry carries [`CteLeaf::SchemaOnly`]: only its columns are read (by
/// `resolve_table`), and the whole set is popped when the guard drops — before Phase 2 —
/// so no schema-only entry ever reaches the Phase-2 leaf builder. CTEs are registered in
/// order (a later CTE sees the earlier ones, matching `compile_with`). A recursive CTE's
/// schema comes from its initial-select (the compound's leftmost arm, which cannot
/// self-reference), so no recursion handling is needed here.
pub(crate) fn push_cte_schemas(
    catalog: &dyn minisqlite_catalog::Catalog,
    with: &With,
) -> Result<CteScopeGuard> {
    let guard = CteScopeGuard::new();
    for cte in &with.ctes {
        let columns = cte_schema(catalog, cte)?;
        push_cte_entry(CteEntry { name: cte.name.clone(), columns, leaf: CteLeaf::SchemaOnly });
    }
    Ok(guard)
}

// ---------------------------------------------------------------------------
// The entrypoint.
// ---------------------------------------------------------------------------

/// Compile a SELECT that carries a leading `WITH` clause, with an optional enclosing
/// scope, into its plan tree and output column names.
///
/// Registers each CTE in order (a later CTE sees the earlier ones; a recursive CTE sees
/// itself as its working table), then compiles the SELECT body against them.
///
/// # The correlated-subquery seam is threaded straight through
///
/// `base_offset` and the correlation collectors (`correlated_out` / `correlated_cols_out`
/// / `nondet_out`) are the SAME seam [`compile_body`] takes, forwarded verbatim: a
/// WITH-bearing SELECT is compiled exactly like a bare one, just with the `WITH`'s CTEs in
/// scope. So a **correlated WITH subquery** — a WITH-bearing subquery whose body references
/// an outer column, e.g. `SELECT t.a, (WITH c(n) AS (VALUES(10)) SELECT n + t.a FROM c)
/// FROM t` — binds its own sources at `base_offset` (the enclosing row's width) and reports
/// its outer references through the collectors, just as a non-WITH correlated subquery does.
/// [`compile_subplan`](crate::compile::select) picks `base_offset` via its trial-at-0 /
/// rebind-at-`outer_width` dance; on the rebind the trial's CTEs were rolled back (the
/// [`Savepoint`](crate::plan_ctx::Savepoint) truncates `ctx.ctes`), so this re-registers
/// them at the IDENTICAL ids. At exec a [`PlanNode::CteScan`] already prepends the `outer`
/// prefix to every row (`ops::cte`), so a CTE reference inside a correlated body lands its
/// columns at `[base_offset, …)` with no CTE-specific exec change.
///
/// The CTE BODIES themselves stay uncorrelated: [`register_cte`] compiles each with no
/// enclosing scope (`parent = None`, base 0), matching SQLite's "temporary view" model
/// where a CTE definition cannot see the enclosing query's columns — an outer reference
/// inside a CTE body is a plain "no such column", not this correlated-body path.
pub fn compile_with(
    ctx: &mut PlanCtx,
    sel: &Select,
    parent: Option<&Scope>,
    base_offset: usize,
    correlated_out: Option<&Cell<bool>>,
    correlated_cols_out: Option<&RefCell<Vec<usize>>>,
    nondet_out: Option<&Cell<bool>>,
) -> Result<(PlanNode, Vec<String>)> {
    let with = sel.with.as_ref().expect("compile_with requires a leading WITH clause");

    // Registered CTEs are visible until this guard drops (end of body compilation, i.e. after
    // the tail `compile_body` returns). Shares the ONE registration path with the DML compilers
    // so the two cannot drift: `register_with` is exactly this `CteScopeGuard` + per-CTE
    // `register_cte` loop, returning the live guard.
    let _guard = register_with(ctx, with)?;

    // Compile the body — the SELECT minus its leading `WITH`. `compile_body` asserts the
    // WITH was already consumed, so hand it a copy with `with: None`; the CTEs stay
    // visible via the scope stack. `parent` + `base_offset` + the collectors are threaded
    // verbatim so a correlated WITH subquery binds its own sources after the outer row and
    // reports its outer references exactly like a non-WITH one; ORDER BY / LIMIT ride along
    // as they belong to the whole query.
    let body = Select {
        with: None,
        body: sel.body.clone(),
        order_by: sel.order_by.clone(),
        limit: sel.limit.clone(),
    };
    compile_body(
        ctx,
        &body,
        parent,
        base_offset,
        correlated_out,
        correlated_cols_out,
        nondet_out,
        None,
    )
}

/// Register a leading `WITH` clause's CTEs, returning a guard that keeps them visible until
/// it drops. This is the ONE CTE-registration path: [`compile_with`] (the SELECT entrypoint)
/// calls it, and so do the INSERT/UPDATE/DELETE compilers — so the two cannot drift.
///
/// A DML statement is not a [`Select`], so it cannot call `compile_with` (which also compiles
/// the SELECT body), but a leading `WITH` on a DML behaves identically to one on a SELECT —
/// per `spec/sqlite-doc/lang_with.html` §1 a CTE "act[s] like a temporary view that exists
/// only for the duration of a single SQL statement", visible to the statement's
/// source/query and its subqueries. So each CTE is compiled into `ctx.ctes` and pushed onto
/// the CTE scope EXACTLY as SELECT does; the DML then compiles its source SELECT (INSERT),
/// or binds its WHERE / SET subqueries (UPDATE/DELETE), and a `FROM c` inside those resolves
/// through the same [`lookup_cte`] path (recursive CTEs included — [`register_cte`] handles
/// the `RecursiveScan` split). The compiled [`CtePlan`]s ride in `ctx.ctes`, which the
/// planner moves into [`Plan::ctes`](crate::plan::Plan::ctes) and the executor materializes
/// before running the DML root — the SAME plan-level attach the SELECT path uses (no new
/// mechanism, and the DML plan nodes are unchanged).
///
/// The caller holds the returned [`CteScopeGuard`] through its own CTE-referencing binding
/// (INSERT source / UPDATE SET+WHERE / DELETE WHERE / RETURNING), then drops it — never
/// leaking a statement's CTEs into a sibling. This uses [`CteScopeGuard`] (a length-restoring
/// pop), NOT [`IsolatedCteScope`] (which SETS the stack aside): the DML's own body MUST see
/// these CTEs, exactly as SELECT's [`compile_with`] does. If `register_cte` fails partway,
/// the locally-held guard drops on the `?` and unwinds the partial registration.
///
/// IMPORTANT — the caller MUST drop this guard BEFORE it calls `compile_triggers`. A trigger
/// body is a SEPARATE program compiled under a fresh [`PlanCtx`] that nonetheless SHARES this
/// thread-local scope, and per `spec/sqlite-doc/lang_with.html` §1 a CTE lives "only for the
/// duration of a single SQL statement" (the firing one) — so it must be invisible inside the
/// triggers that statement fires. (Before WITH-on-DML existed, a top-level DML always started
/// from an empty scope, so trigger compilation never saw a CTE; that is no longer automatic,
/// hence the explicit `drop` in each DML compiler.) Dropping pops only the name->id scope
/// entries — `ctx.ctes` (-> [`Plan::ctes`](crate::plan::Plan::ctes)) is untouched, so the DML
/// body's own `CteScan` ids stay valid. The root-cause defense (isolating the scope inside
/// `compile_action` itself, as [`crate::compile::view`] does for view bodies) belongs to the
/// trigger compiler; see `conformance_dml_cte_trigger` for the regression this drop closes.
pub(crate) fn register_with(ctx: &mut PlanCtx, with: &With) -> Result<CteScopeGuard> {
    let guard = CteScopeGuard::new();
    for cte in &with.ctes {
        register_cte(ctx, cte)?;
    }
    Ok(guard)
}

// ---------------------------------------------------------------------------
// Registering one CTE.
// ---------------------------------------------------------------------------

/// Register one CTE into `ctx.ctes` and the CTE scope: compute its output schema
/// (honoring an explicit `WITH t(a,b) AS …` column list), compile its body, and record
/// a scope entry mapping its name to the registered plan.
///
/// A CTE is recursive iff a compound arm of its body references the CTE's own name in a
/// FROM clause (the `RECURSIVE` keyword neither forces nor is required — spec §2/§5).
/// A recursive CTE compiles to a [`CtePlan::Recursive`]; every other CTE (including a
/// non-self-referential one under `WITH RECURSIVE`) to a [`CtePlan::Materialized`].
fn register_cte(ctx: &mut PlanCtx, cte: &Cte) -> Result<()> {
    let columns = cte_schema(ctx.catalog, cte)?;
    let column_count = columns.len();

    match split_recursive(&cte.select, &cte.name)? {
        Some((seed_body, step_body, union_all)) => {
            // A LIMIT/OFFSET on a recursive CTE body bounds the rows of the recursive
            // table (spec §3). SQLite's ORDER BY on that body reorders the recursion QUEUE
            // (which rows recurse first), which a breadth-first fixpoint can't reproduce —
            // and with a LIMIT that order decides which rows survive. So LIMIT + ORDER BY
            // together can't be honored correctly; reject it loud rather than return wrong
            // rows. (An ORDER BY without LIMIT is a no-op on the unordered CTE result and is
            // dropped silently.) LIMIT/OFFSET alone is applied below via a materialized
            // wrapper.
            if cte.select.limit.is_some() && !cte.select.order_by.is_empty() {
                return Err(Error::sql(format!(
                    "ORDER BY with LIMIT on the recursive common table expression \"{}\" \
                     is not supported",
                    cte.name
                )));
            }

            // The initial-select(s): the CTE name is NOT yet in scope, so an
            // initial-select that referenced it would be a "no such table" error — which
            // matches the rule that only the recursive-select may reference the table.
            let (seed, seed_names) = compile_arm_body(ctx, seed_body)?;
            check_arm_width(&cte.name, seed_names.len(), column_count)?;

            // The recursive-select(s): compile with the CTE visible as its own working
            // table (a self-reference lowers to `RecursiveScan`). The temporary entry is
            // removed again before the real (`CteScan`) entry is pushed for the body.
            let restore = cte_scope_len();
            push_cte_entry(CteEntry {
                name: cte.name.clone(),
                columns: columns.clone(),
                leaf: CteLeaf::Recursive,
            });
            let step_result = compile_arm_body(ctx, step_body);
            truncate_cte_scope(restore);
            let (step, step_names) = step_result?;
            check_arm_width(&cte.name, step_names.len(), column_count)?;

            let rec_id = ctx.ctes.len();
            ctx.ctes.push(CtePlan::Recursive {
                name: cte.name.clone(),
                column_count,
                seed,
                step,
                union_all,
            });

            // A body LIMIT/OFFSET is honored by a second, materialized CTE that scans the
            // raw fixpoint (`CteScan`) under a `LIMIT`/`OFFSET`. The executor produces the
            // fixpoint LAZILY in seed-then-round order and the scan streams it, so the
            // wrapper's LIMIT pulls only the rows it needs and then stops — which stops the
            // recursion early. That bounds even a NON-terminating body (the
            // `... UNION ALL ... LIMIT n` idiom) without materializing the whole fixpoint or
            // reaching the executor's safety cap. `OFFSET` rows are still generated (they
            // seed later rows) but skipped from the output — matching SQLite's during-
            // recursion bound (spec §3), which produces rows in that same order. Only the
            // wrapper is referenced by name; the raw fixpoint is scanned once, by it.
            let leaf_id = match &cte.select.limit {
                None => rec_id,
                Some(limit) => {
                    let scan = PlanNode::CteScan { id: rec_id, column_count };
                    let body = build_limit(ctx, scan, limit)?;
                    let wrap_id = ctx.ctes.len();
                    ctx.ctes.push(CtePlan::Materialized {
                        name: cte.name.clone(),
                        column_count,
                        body,
                    });
                    wrap_id
                }
            };
            push_cte_entry(CteEntry {
                name: cte.name.clone(),
                columns,
                leaf: CteLeaf::Scan(leaf_id),
            });
        }
        None => {
            // An ordinary CTE: compile the full body (its own ORDER BY / LIMIT and any
            // nested `WITH` included). `compile_select` re-enters the WITH dispatch for a
            // nested `WITH`, extending the scope stack for the inner body only.
            let (body, names) = compile_select(ctx, &cte.select)?;
            // Phase-1 schema width (`derived_schema`) must equal the compiled body width;
            // any explicit column-list arity mismatch was already caught by `cte_schema`,
            // so a disagreement here is an internal derived-vs-compiled bug, not user error.
            if names.len() != column_count {
                return Err(Error::sql(format!(
                    "internal error: common table expression \"{}\" schema has {} columns \
                     but its body compiled to {}",
                    cte.name,
                    column_count,
                    names.len()
                )));
            }
            let id = ctx.ctes.len();
            ctx.ctes.push(CtePlan::Materialized {
                name: cte.name.clone(),
                column_count,
                body,
            });
            push_cte_entry(CteEntry { name: cte.name.clone(), columns, leaf: CteLeaf::Scan(id) });
        }
    }
    Ok(())
}

/// The CTE's output schema: the body's derived schema, then the explicit column-name list
/// (`WITH t(a,b) AS …`) applied — renaming the columns and enforcing the arity SQLite
/// requires. The affinity/collation are carried through from the derived schema, which
/// [`crate::compile::from::derived_schema`] computes per datatype3 §3.3 (a bare column ref
/// inherits its column's affinity + collation, a `CAST` takes the cast type's, any other
/// expression is NONE + BINARY) — the explicit column list only renames, never retypes.
fn cte_schema(catalog: &dyn minisqlite_catalog::Catalog, cte: &Cte) -> Result<Vec<SynthCol>> {
    let base = derived_schema(catalog, &cte.select)?;
    match &cte.columns {
        None => Ok(base),
        Some(list) => {
            if list.len() != base.len() {
                return Err(Error::sql(format!(
                    "table {} has {} values for {} columns",
                    cte.name,
                    base.len(),
                    list.len()
                )));
            }
            Ok(list
                .iter()
                .zip(base)
                .map(|(name, col)| SynthCol {
                    name: name.clone(),
                    affinity: col.affinity,
                    collation: col.collation,
                    hidden: false,
                })
                .collect())
        }
    }
}

/// Compile one reconstructed recursion arm (seed or step) as a standalone body: no
/// enclosing scope (a CTE body never correlates to the outer query), register base 0, and
/// no ORDER BY / LIMIT of its own (those belong to the whole compound, and for a
/// recursive CTE bound the recursion — carried by the executor, not the arm).
fn compile_arm_body(ctx: &mut PlanCtx, body: SelectBody) -> Result<(PlanNode, Vec<String>)> {
    let sel = Select { with: None, body, order_by: Vec::new(), limit: None };
    compile_body(ctx, &sel, None, 0, None, None, None, None)
}

/// A recursion arm's compiled width must equal the CTE's declared width — SQLite's
/// compound-arity rule that the initial-select(s) and recursive-select(s) all have the
/// same number of result columns.
fn check_arm_width(name: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::sql(format!(
            "SELECTs to the left and right of a compound in the CTE \"{name}\" \
             do not have the same number of result columns"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Recursion detection: split a compound body into seed / step.
// ---------------------------------------------------------------------------

/// If `select`'s body is a compound whose arms split into leading initial-select(s) and
/// trailing recursive-select(s) — the latter referencing `name` in a FROM clause — return
/// `Ok(Some((seed_body, step_body, union_all)))`; `Ok(None)` for an ordinary CTE; and an
/// `Err` for a self-referential body that is not a well-formed recursion (SQLite rejects
/// these, so failing loud beats compiling a silently-wrong plan).
///
/// The arms are flattened left-to-right; `r` is the first arm that references the CTE.
/// Everything before `r` is the seed (initial-selects), everything from `r` the step
/// (recursive-selects), and `union_all` is the operator joining the two (spec §3 makes
/// that operator — `UNION` vs `UNION ALL` — decide whole-run dedup). `r == 0` (a leading
/// recursive-select, i.e. no initial-select) is not a valid split and yields `Ok(None)`,
/// so the ordinary path then reports the self-reference as an unresolved table.
///
/// Two malformed-recursion shapes are rejected here rather than mis-compiled:
/// - the seed↔step connector is `INTERSECT`/`EXCEPT` (SQLite requires `UNION`/`UNION ALL`
///   between the initial- and recursive-selects), and
/// - a single recursive-select references the CTE more than once (e.g. `FROM c a JOIN c b`);
///   SQLite errors "recursive table may not appear more than once", whereas emitting two
///   `RecursiveScan`s would cross-product the working table into wrong rows.
fn split_recursive(select: &Select, name: &str) -> Result<Option<(SelectBody, SelectBody, bool)>> {
    let Some((arms, ops)) = flatten_compound(&select.body) else {
        return Ok(None);
    };
    let Some(r) = arms.iter().position(|core| core_references(core, name)) else {
        return Ok(None);
    };
    if r == 0 {
        return Ok(None);
    }

    // Each recursive-select (arms `r..`) may reference the working table at most once.
    for core in &arms[r..] {
        if count_core_references(core, name) > 1 {
            return Err(Error::sql(format!(
                "recursive table \"{name}\" may not appear more than once in a recursive-select"
            )));
        }
    }

    let union_all = match ops[r - 1] {
        CompoundOp::UnionAll => true,
        CompoundOp::Union => false,
        CompoundOp::Intersect | CompoundOp::Except => {
            return Err(Error::sql(format!(
                "recursive table \"{name}\" must be joined to its initial-select with \
                 UNION or UNION ALL, not INTERSECT or EXCEPT"
            )));
        }
    };
    let seed = rebuild_body(&arms, &ops, 0, r);
    let step = rebuild_body(&arms, &ops, r, arms.len());
    Ok(Some((seed, step, union_all)))
}

/// Flatten a (left-associative) compound body into its arms in source order and the
/// operators between them (`ops[i]` joins `arms[i]` and `arms[i+1]`). `None` for a
/// non-compound body (a single core cannot be recursive — it has no UNION arm).
fn flatten_compound(body: &SelectBody) -> Option<(Vec<&SelectCore>, Vec<CompoundOp>)> {
    if let SelectBody::Select(_) = body {
        return None;
    }
    let mut arms = Vec::new();
    let mut ops = Vec::new();
    collect_arms(body, &mut arms, &mut ops);
    Some((arms, ops))
}

fn collect_arms<'a>(
    body: &'a SelectBody,
    arms: &mut Vec<&'a SelectCore>,
    ops: &mut Vec<CompoundOp>,
) {
    match body {
        SelectBody::Select(core) => arms.push(core),
        SelectBody::Compound { op, left, right } => {
            collect_arms(left, arms, ops);
            ops.push(*op);
            arms.push(right);
        }
    }
}

/// Rebuild a left-deep compound `SelectBody` from `arms[lo..hi]` (joined by
/// `ops[lo..hi-1]`), cloning the cores. A single arm is a bare `Select` body.
fn rebuild_body(arms: &[&SelectCore], ops: &[CompoundOp], lo: usize, hi: usize) -> SelectBody {
    let mut body = SelectBody::Select(arms[lo].clone());
    for k in (lo + 1)..hi {
        body = SelectBody::Compound {
            op: ops[k - 1],
            left: Box::new(body),
            right: arms[k].clone(),
        };
    }
    body
}

/// Whether a select core references the CTE `name` in its FROM clause (an unqualified
/// table reference, possibly inside a join). A subquery / table-valued function in FROM
/// is a separate scope where SQLite disallows the recursive self-reference, so it is not
/// descended into.
fn core_references(core: &SelectCore, name: &str) -> bool {
    count_core_references(core, name) > 0
}

/// How many times a select core references the CTE `name` in its FROM clause. Used to
/// reject a recursive-select that names the working table more than once (SQLite errors;
/// two `RecursiveScan`s would silently cross-product). Same scoping rule as
/// [`core_references`]: a subquery / table-valued function in FROM is not descended into.
fn count_core_references(core: &SelectCore, name: &str) -> usize {
    match core {
        SelectCore::Values(_) => 0,
        SelectCore::Query { from, .. } => {
            from.as_ref().map_or(0, |tree| count_tree_references(tree, name))
        }
    }
}

fn count_tree_references(tree: &JoinTree, name: &str) -> usize {
    match tree {
        JoinTree::Table(tos) => count_tos_references(tos, name),
        JoinTree::Join { left, right, .. } => {
            count_tree_references(left, name) + count_tos_references(right, name)
        }
    }
}

fn count_tos_references(tos: &TableOrSubquery, name: &str) -> usize {
    match tos {
        TableOrSubquery::Table { name: qn, .. } => {
            usize::from(qn.schema.is_none() && qn.name.eq_ignore_ascii_case(name))
        }
        TableOrSubquery::Join(tree) => count_tree_references(tree, name),
        TableOrSubquery::Subquery { .. } | TableOrSubquery::TableFunction { .. } => 0,
    }
}
