//! The compiled expression IR: [`EvalExpr`] and its helper op/metadata types.
//!
//! Every operand that would require a per-row lookup in an AST walker is resolved
//! to O(1) data here: a column is a register index, a function is a resolved
//! `Arc<dyn ScalarFunction>` handle, and comparison affinity + collation are baked
//! into a [`CompareMeta`] at bind time. The op enums are this crate's *own* (they
//! intentionally do not reuse the SQL AST's), so this crate never depends on the
//! grammar.
//!
//! `EvalExpr` derives `Clone` (cheap: `Box`/`Arc`/`Vec` clones) and `Debug`, but
//! *not* `PartialEq` — a resolved function handle behind `dyn` is not comparable,
//! and structural expression equality is not a concept the evaluator needs.

use std::sync::Arc;

use minisqlite_types::{Affinity, Collation, Value};

use crate::function::ScalarFunction;

/// Identifies a subquery whose plan the executor holds; the evaluator calls back
/// through [`crate::EvalContext`] with this id rather than embedding the subplan
/// (which would drag the plan/exec crates into this AST-free crate).
pub type SubqueryId = usize;

/// Precomputed comparison metadata for one binary comparison, produced by the
/// binder so the per-row evaluator never recomputes affinity or collation.
///
/// `apply_left`/`apply_right` are the affinity to apply to each operand *before*
/// comparing (`None` = leave the operand untouched). `collation` is the resolved
/// collating sequence for the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompareMeta {
    pub apply_left: Option<Affinity>,
    pub apply_right: Option<Affinity>,
    pub collation: Collation,
}

/// Prefix unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-x` (arithmetic negation).
    Neg,
    /// `+x` (identity — returns the operand completely untouched, no coercion).
    Pos,
    /// `NOT x` (three-valued logical negation).
    Not,
    /// `~x` (bitwise complement).
    BitNot,
}

/// Arithmetic operators (`+ - * / %`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Integer bitwise/shift operators (`& | << >>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOp {
    And,
    Or,
    Shl,
    Shr,
}

/// Comparison operators. Equality/inequality double as the `IS`/`IS NOT` operators
/// when [`EvalExpr::Compare`]`::null_safe` is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Which pattern-matching operator a [`EvalExpr::Like`] node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LikeKind {
    /// `LIKE`: `%`/`_` wildcards, ASCII-case-insensitive, optional `ESCAPE`.
    Like,
    /// `GLOB`: `*`/`?`/`[...]` wildcards, case-sensitive.
    Glob,
}

/// Which `CURRENT_*` keyword a [`EvalExpr::Now`] node renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NowKind {
    /// `CURRENT_DATE` -> `YYYY-MM-DD`.
    Date,
    /// `CURRENT_TIME` -> `HH:MM:SS`.
    Time,
    /// `CURRENT_TIMESTAMP` -> `YYYY-MM-DD HH:MM:SS`.
    Timestamp,
}

/// One `WHEN … THEN …` arm of a [`EvalExpr::Case`].
///
/// `cmp` distinguishes the two CASE forms: `Some(meta)` is a *simple* CASE, where
/// `when` is compared for equality against the CASE operand using `meta`;
/// `None` is a *searched* CASE, where `when` is evaluated as a boolean condition.
#[derive(Debug, Clone)]
pub struct CaseWhen {
    pub when: EvalExpr,
    pub cmp: Option<CompareMeta>,
    pub then: EvalExpr,
}

/// A bound, ready-to-evaluate scalar expression.
///
/// The evaluator ([`crate::eval`]) walks this tree once per row. Sub-expressions
/// are boxed; lists are `Vec`s; a function is an `Arc` handle. See each variant's
/// note for its evaluation contract (the authoritative rules are implemented in
/// `eval.rs` and pinned by that module's tests).
#[derive(Debug, Clone)]
pub enum EvalExpr {
    /// A constant value.
    Literal(Value),
    /// Read register slot `usize` from the current row (`regs[i]`).
    Column(usize),
    /// Read bind-parameter `usize` via [`crate::EvalContext::param`].
    Param(usize),
    /// `CURRENT_DATE` / `CURRENT_TIME` / `CURRENT_TIMESTAMP`.
    Now(NowKind),
    /// A prefix unary operation.
    Unary { op: UnaryOp, operand: Box<EvalExpr> },
    /// An arithmetic operation.
    Arith { op: ArithOp, left: Box<EvalExpr>, right: Box<EvalExpr> },
    /// String concatenation (`||`).
    Concat { left: Box<EvalExpr>, right: Box<EvalExpr> },
    /// An integer bitwise/shift operation.
    Bitwise { op: BitOp, left: Box<EvalExpr>, right: Box<EvalExpr> },
    /// A comparison. `null_safe` selects `IS`/`IS NOT` semantics (never yields
    /// NULL): with `null_safe`, `op` is [`CmpOp::Eq`] for `IS` and [`CmpOp::Ne`]
    /// for `IS NOT`.
    Compare {
        op: CmpOp,
        null_safe: bool,
        left: Box<EvalExpr>,
        right: Box<EvalExpr>,
        meta: CompareMeta,
    },
    /// Three-valued `AND`.
    And(Box<EvalExpr>, Box<EvalExpr>),
    /// Three-valued `OR`.
    Or(Box<EvalExpr>, Box<EvalExpr>),
    /// `x ISNULL` / `x IS NULL` (returns 1/0, never NULL).
    IsNull(Box<EvalExpr>),
    /// `x NOTNULL` / `x IS NOT NULL` (returns 1/0, never NULL).
    NotNull(Box<EvalExpr>),
    /// `subject [NOT] BETWEEN low AND high`. `subject` is evaluated once.
    Between {
        negated: bool,
        subject: Box<EvalExpr>,
        low: Box<EvalExpr>,
        high: Box<EvalExpr>,
        low_meta: CompareMeta,
        high_meta: CompareMeta,
    },
    /// `subject [NOT] IN (items…)` with a literal value list.
    InList { negated: bool, subject: Box<EvalExpr>, items: Vec<EvalExpr>, meta: CompareMeta },
    /// `subject [NOT] IN (subquery)`.
    InSubquery { negated: bool, subject: Box<EvalExpr>, id: SubqueryId, meta: CompareMeta },
    /// `(subjects…) [NOT] IN (subquery)` — a row-value (tuple) IN-subquery probe
    /// (rowvalue.html §2.2). `subjects` are the per-element bound LHS expressions and
    /// `metas` the per-element comparison metadata; both are the same width as the
    /// subquery's output. Evaluated via [`crate::EvalContext::eval_in_subquery_row`],
    /// which the scalar [`InSubquery`](EvalExpr::InSubquery) generalizes to a tuple.
    InSubqueryRow {
        negated: bool,
        subjects: Vec<EvalExpr>,
        id: SubqueryId,
        metas: Vec<CompareMeta>,
    },
    /// `[NOT] EXISTS (subquery)`.
    Exists { negated: bool, id: SubqueryId },
    /// `(subquery)` used as a scalar (first row, first column; NULL if no rows).
    ScalarSubquery(SubqueryId),
    /// Column `col` of the FIRST row of subplan `id` (NULL if the subquery returns no
    /// row). The row-value generalization of [`ScalarSubquery`](EvalExpr::ScalarSubquery),
    /// which is `col == 0`: a column-list `UPDATE` source `SET (a, b, …) = (SELECT x, y,
    /// … FROM …)` (rowvalue.html §2.3) emits ONE `ScalarSubqueryColumn` per target
    /// column, all sharing the one subplan `id`, so the subquery runs once per (outer)
    /// row and each assignment reads its positional column of that single result row.
    ScalarSubqueryColumn { id: SubqueryId, col: usize },
    /// `COALESCE(items…)` — first non-NULL argument, short-circuiting.
    Coalesce(Vec<EvalExpr>),
    /// `NULLIF(left, right)` — NULL if the two compare equal, else `left`.
    NullIf { left: Box<EvalExpr>, right: Box<EvalExpr>, meta: CompareMeta },
    /// A `CASE` expression (simple if `operand` is `Some`, else searched).
    Case { operand: Option<Box<EvalExpr>>, whens: Vec<CaseWhen>, else_expr: Option<Box<EvalExpr>> },
    /// `CAST(operand AS <type>)` reduced to a target affinity.
    Cast { affinity: Affinity, operand: Box<EvalExpr> },
    /// `operand COLLATE <name>` — pass-through at eval time; the collation is
    /// already baked into the enclosing comparison/sort metadata by the binder.
    Collate { collation: Collation, operand: Box<EvalExpr> },
    /// `subject [NOT] LIKE/GLOB pattern [ESCAPE escape]`.
    Like {
        negated: bool,
        kind: LikeKind,
        subject: Box<EvalExpr>,
        pattern: Box<EvalExpr>,
        escape: Option<Box<EvalExpr>>,
    },
    /// A scalar function call against a resolved handle.
    Func { func: Arc<dyn ScalarFunction>, args: Vec<EvalExpr> },
    /// `RAISE(action[, message])` — only meaningful inside a trigger body
    /// (lang_createtrigger.html §6). `ABORT`/`FAIL`/`ROLLBACK` carry a `message` and
    /// raise a constraint error when evaluated (aborting the statement, which the
    /// engine's implicit transaction rolls back); `IGNORE` carries no message and is a
    /// non-error control signal ("abandon the current row's operation, continue the
    /// statement") — the evaluator routes it through `EvalContext::signal_raise_ignore`
    /// so the enclosing trigger fire turns it into a row-skip (see the evaluator arm).
    Raise { kind: RaiseKind, message: Option<String> },
}

/// The action of a `RAISE(...)` trigger expression (lang_createtrigger.html §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaiseKind {
    /// `RAISE(ABORT, msg)` — abort the current statement and undo its changes.
    Abort,
    /// `RAISE(FAIL, msg)` — fail the statement (prior changes are not undone).
    Fail,
    /// `RAISE(ROLLBACK, msg)` — roll back the whole transaction.
    Rollback,
    /// `RAISE(IGNORE)` — abandon the current row's trigger program without an error.
    Ignore,
}

// Compile-time contract: the compiled IR must stay `Send + Sync` so a prepared
// statement can be shared immutably across WAL reader threads (each thread holds its
// own `&mut EvalContext`; the evaluator has no shared writable state). This holds
// today because `EvalExpr` contains only `Value`/`Box`/`Vec`/`Arc<dyn Send+Sync>`.
// Adding a non-`Sync` field (an `Rc`, a `Cell`, or a `dyn Trait` lacking `Send+Sync`)
// would silently drop the auto-trait and break that sharing — this turns that
// regression into a compile error right here. The closure is never called; coercing
// it to `fn()` is what forces its body (and thus the bounds) to type-check.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EvalExpr>();
    assert_send_sync::<CompareMeta>();
    assert_send_sync::<CaseWhen>();
};
