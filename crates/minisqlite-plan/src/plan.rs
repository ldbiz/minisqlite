//! The plan vocabulary: the [`Plan`] a statement compiles to and the [`PlanNode`]
//! operator tree the executor runs. Data only — the compiler builds these, the
//! executor consumes them; there is no behavior here.
//!
//! # ROW / REGISTER CONVENTION (the shared contract)
//!
//! This is the interface both the query compiler (which produces plans) and the
//! executor (which runs them) bind against. Every rule below is load-bearing: a
//! plan is only correct if the compiler emits column references consistent with the
//! row layout each operator produces, and the executor materializes exactly that
//! layout.
//!
//! * A row flowing between operators is a `Vec<Value>` (= [`minisqlite_types::Row`]).
//!   [`EvalExpr::Column(i)`](minisqlite_expr::EvalExpr::Column) reads `row[i]`.
//! * Base-table scan of a table with `N` columns (`CREATE TABLE` order) emits width
//!   `N+1`: `[c0, c1, …, c_{N-1}, rowid]`; register `N` holds the rowid as
//!   `Value::Integer`. A stored record with fewer than `N` columns is padded
//!   (missing trailing columns take their default / NULL). If a column is
//!   `INTEGER PRIMARY KEY` it aliases the rowid (a stored NULL is replaced by the
//!   rowid). Column refs bind: a named column → its 0-based schema index;
//!   `rowid` / `_rowid_` / `oid` → register `N`.
//! * `Join(left width WL, right width WR)` emits `WL+WR`: left occupies `[0, WL)`,
//!   right occupies `[WL, WL+WR)`; the compiler binds right-side column refs to
//!   `WL+j`. For `LEFT` / `RIGHT` / `FULL OUTER`, the unmatched side's registers are
//!   all `Value::Null`.
//! * `Aggregate` emits `[group_key_0 … group_key_{G-1}, agg_result_0 … agg_result_{A-1}]`;
//!   `HAVING` and the post-aggregate projection bind against this. With no `GROUP BY`
//!   over an EMPTY input the operator still emits exactly ONE row (e.g. `count = 0`,
//!   `sum = NULL`). When the single-min/max bare-column special case applies
//!   ([`Aggregate::minmax_bare`], lang_select.html §2.5) the operator APPENDS the
//!   captured bare-column values after the results, so the row is
//!   `[group_keys.., agg_results.., captured_bare..]` and a bare result column binds
//!   to `num_keys + aggregates.len() + slot`.
//! * `Project` emits width = number of projected exprs. `Filter` / `Sort` / `Limit`
//!   / `Distinct` pass the row through unchanged (same layout as their input).
//! * `SetOp`: both inputs must be the same width; output width = that; result column
//!   NAMES come from the left input. `Values`: width = exprs per row (all rows equal
//!   width). `SingleRow`: one row, zero columns.
//! * A correlated subplan's [`EvalExpr::Column(i)`](minisqlite_expr::EvalExpr::Column)
//!   for `i` in `[0, outer_width)` refers to the OUTER row handed to the
//!   `EvalContext` subquery callback; `i >= outer_width` refers to the subquery's own
//!   operator rows.
//! * When a write fires a trigger, the firing executor supplies the target table's OLD
//!   row in registers `[0, W)` and its NEW row in `[W, 2W)` (`W = column_count + 1`) as
//!   the leading correlated-style registers, so a trigger's action / `WHEN` binds
//!   `OLD`/`NEW` there and the action's own operators start at register `2W`. The full
//!   rule is the binding contract on [`TriggerProgram`].
//! * Widths (`column_count`, `left_width`, `right_width`, `source_width`) are carried
//!   EXPLICITLY in nodes so the executor reads them as data rather than re-deriving.
//! * `column_count` has two contracts by node, so read the node's own doc: on a
//!   base-table node (`SeqScan`, `RowidScan`, `IndexScan`, `Insert`, `Update`,
//!   `Delete`) it is `N`, the table's column count, and the row width is `N+1` (the
//!   trailing rowid register); on a CTE / derived node (`CteScan`, `RecursiveScan`,
//!   `CtePlan`) it is the output width directly, with NO trailing rowid register.

use minisqlite_catalog::PragmaFunction;
use minisqlite_expr::{AggregateCall, EvalExpr, SortKey};
use minisqlite_functions::JsonTableKind;
use minisqlite_types::{Affinity, Collation, DbIndex};

use crate::access::{IndexScan, RowidScan, SeqScan};
use crate::window::{WindowFrame, WindowFuncKind};

/// A compiled statement: the operator tree plus the side tables and flags the
/// executor needs to run it. This is the whole output of the planning seam for one
/// statement.
#[derive(Debug, Clone)]
pub struct Plan {
    /// The top of the operator tree; pulling rows from it produces the result.
    pub root: PlanNode,
    /// Output column names in result order (for `QueryResult.columns` / `RETURNING`).
    /// Empty for pure no-row DML/DDL.
    pub result_columns: Vec<String>,
    /// Materialized subplans (`WITH` CTEs / derived tables), referenced by
    /// [`PlanNode::CteScan`]`{ id }` (`id` indexes this vector).
    pub ctes: Vec<CtePlan>,
    /// Expression subqueries (scalar / `EXISTS` / `IN`-select), referenced by
    /// `EvalExpr` subquery ids ([`minisqlite_expr::SubqueryId`], an index into this
    /// vector).
    pub subqueries: Vec<SubPlan>,
    /// `true` for statements that change schema or data (DML/DDL) so the engine wraps
    /// a write transaction; `false` for pure reads.
    pub mutates: bool,
    /// GENERATED-column programs, one [`TableGenerated`] per base table THIS plan touches
    /// that has any generated column (`gencol.html`), populated after compilation by
    /// [`crate::compile::generated::populate_generated`]. Empty for a plan over no
    /// generated-column table (the common case — the read/write fast paths do nothing).
    ///
    /// It lives on the `Plan` (not on the scan-leaf nodes or the `Insert`/`Update` nodes)
    /// so a base table's generation expressions are bound and carried ONCE per statement,
    /// reachable by every operator through the shared [`crate::plan::Plan`] the executor's
    /// `Env` holds — the same way `subqueries`/`ctes` ride here. The scan leaves compute
    /// VIRTUAL columns on read; the `Insert`/`Update` executors compute every generated
    /// column on write and omit VIRTUAL ones from the stored record. A NESTED trigger
    /// action runs under its OWN `Plan`, whose `generated` is populated by the same
    /// recursive walk, so each action reaches its target's programs through its own plan.
    pub generated: Vec<TableGenerated>,
}

impl Plan {
    /// The generated-column programs (STORED + VIRTUAL, in column order) for the base table
    /// `table` IN NAMESPACE `db`, or `&[]` when that table has none / is not touched by this
    /// plan. Matched case-insensitively (SQLite folds identifiers over ASCII). The scan
    /// leaves filter to the VIRTUAL subset on read; the write operators use the whole list.
    ///
    /// `db` is part of the key (not the name alone): a temp/attached table can SHADOW a
    /// same-named `main` table with DIFFERENT generated columns, so a lookup by name only
    /// would apply the wrong namespace's programs to a `main.t` write/read (a wrong
    /// STORED/VIRTUAL value, or an out-of-range column index when the arities differ). Every
    /// caller passes the node's own resolved `db` ([`SeqScan::db`](crate::access::SeqScan::db)
    /// / [`Insert::db`] / …), the same namespace the row is opened on. With no temp/attach
    /// `db == MAIN` for every table, so this is the same match a name-only key would make.
    pub fn generated_programs(&self, db: DbIndex, table: &str) -> &[GeneratedProgram] {
        self.generated
            .iter()
            .find(|t| t.db == db && t.table.eq_ignore_ascii_case(table))
            .map(|t| t.programs.as_slice())
            .unwrap_or(&[])
    }
}

/// The bound generation program for ONE generated column: its schema column index, the
/// generation expression bound to an [`EvalExpr`] over the table's logical row
/// `[c0..c_{N-1}, rowid]` (an INTEGER PRIMARY KEY reference resolves to the trailing
/// rowid register `N`, exactly as CHECK/DEFAULT bind), whether it is STORED, and the
/// column's affinity (applied to the computed value, since a generated column's datatype
/// is fixed by its declared type — `gencol.html`).
///
/// Ordered within a [`TableGenerated`] in DEPENDENCY order (by
/// `compile::generated::bind_generated_programs`): a program appears only after every
/// generated column its expression references, so the read/write compute steps — which
/// iterate the list in order — fill each referenced register before it is read
/// (`gencol.html` §2.2: a generated column may reference any other, in any declaration
/// order, cycles excluded).
#[derive(Debug, Clone)]
pub struct GeneratedProgram {
    /// The generated column's index in `CREATE TABLE` order (`0..N`).
    pub col_index: usize,
    /// The generation expression, bound over the logical row `[c0..c_{N-1}, rowid]`.
    pub expr: EvalExpr,
    /// `true` for `STORED` (materialized in the record), `false` for `VIRTUAL`
    /// (computed on read, never stored).
    pub stored: bool,
    /// The column's affinity, applied to the computed value before it is stored / read.
    pub affinity: Affinity,
}

impl GeneratedProgram {
    /// Whether this program is for a VIRTUAL generated column (`stored == false`) — i.e. one
    /// computed on every read and never written to the record. This is the program-side mirror
    /// of [`minisqlite_catalog::ColumnDef::is_virtual_generated`]: the same "does it take a
    /// stored slot?" invariant, named on both the schema column and its bound program so the
    /// two cannot drift. Read/scan cursors key `has_virtual` on this (`… .any(|p| p.is_virtual())`).
    pub fn is_virtual(&self) -> bool {
        !self.stored
    }
}

/// The generated-column programs for one base table, keyed by its namespace AND name.
/// Carried on [`Plan::generated`]; looked up by [`Plan::generated_programs`]. `db` is part
/// of the key because a temp/attached table can SHADOW a same-named `main` table with
/// DIFFERENT generated columns — keying by name alone would bind one namespace's programs to
/// the other's write/read. With no temp/attach every `db` is `MAIN`, so a keyed lookup makes
/// the same match a name-only key would.
#[derive(Debug, Clone)]
pub struct TableGenerated {
    /// The namespace the base table lives in (`main`/`temp`/attached).
    pub db: DbIndex,
    /// The base table's name (matched case-insensitively).
    pub table: String,
    /// Every generated column's program, in `CREATE TABLE` column order.
    pub programs: Vec<GeneratedProgram>,
}

/// One operator in the plan tree. Children are boxed so the tree is a single owned
/// value. Each variant's doc states the row shape it emits, per the module's
/// ROW/REGISTER convention.
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Full table scan, ascending rowid. Emits `[cols…, rowid]` (width `N+1`).
    SeqScan(SeqScan),
    /// Rowid point-eq or rowid-range access path on the table b-tree (the
    /// `INTEGER PRIMARY KEY` / rowid path). Emits `[cols…, rowid]` (width `N+1`).
    RowidScan(RowidScan),
    /// Index access path: seek/scan the index b-tree, then (unless `covering`) fetch
    /// each row from the table by the trailing rowid. Emits `[cols…, rowid]`
    /// (width `N+1`).
    IndexScan(IndexScan),
    /// Keep rows where `predicate` is TRUE. Three-valued: a NULL or FALSE result
    /// drops the row. Passes the row through unchanged (input layout).
    Filter { input: Box<PlanNode>, predicate: EvalExpr },
    /// Evaluate `exprs` against the input row and emit them as the new row. Output
    /// width = `exprs.len()`.
    Project { input: Box<PlanNode>, exprs: Vec<EvalExpr> },
    /// Order the input by `keys` (each a bound key expr plus its sort modifiers).
    /// Passes rows through unchanged (input layout). `limit`, when `Some`, is the
    /// retention bound the enclosing `LIMIT`/`OFFSET` places on this sort: the operator
    /// keeps only the `offset + limit` rows a full sort would place first (a bounded
    /// top-k) instead of materializing the whole input. It is a pure optimization — the
    /// emitted rows are byte-identical to the full stable sort, and the
    /// [`Limit`](PlanNode::Limit) node above still applies the real skip/take. `None` is a
    /// full stable sort (no `LIMIT`, or a bound the sort cannot safely use). See
    /// [`SortLimit`].
    Sort { input: Box<PlanNode>, keys: Vec<SortKey>, limit: Option<SortLimit> },
    /// `LIMIT`/`OFFSET`: emit at most `limit` rows after skipping `offset`. Both are
    /// expressions (SQLite allows arbitrary scalar expressions there), evaluated once
    /// before iteration. `None` means no limit / no offset. Passes rows through
    /// unchanged (input layout).
    Limit { input: Box<PlanNode>, limit: Option<EvalExpr>, offset: Option<EvalExpr> },
    /// `DISTINCT`: emit each distinct whole row once, comparing by storage-class
    /// equality with per-column collation. Passes rows through unchanged (input
    /// layout).
    ///
    /// `column_collations` gives the collating sequence for each output column's
    /// duplicate comparison (same length as the row), so a `DISTINCT` over a
    /// `NOCASE`/`RTRIM` column (or an explicit `COLLATE`) folds like SQLite's
    /// `IS DISTINCT FROM` (lang_select.html §2.6, datatype3.html §7.1) rather than
    /// under BINARY. Resolved once at compile time from each output column's source
    /// expression (`compile::select`), never per row.
    Distinct { input: Box<PlanNode>, column_collations: Vec<Collation> },
    /// Grouping / aggregation. Emits `[group_keys…, agg_results…]`; see [`Aggregate`].
    Aggregate(Aggregate),
    /// The MIN/MAX index/rowid seek fast path (optoverview.html §13): a single
    /// `MIN(col)`/`MAX(col)` over a whole table satisfied by ONE b-tree seek to the
    /// extremum entry instead of a full-scan aggregate. Emits EXACTLY ONE width-1 row
    /// `[value]` — byte-identical to the aggregate it replaces (whose no-`GROUP BY`,
    /// single-aggregate output is likewise one width-1 row). See [`MinMaxSeek`].
    MinMaxSeek(MinMaxSeek),
    /// A join of two subtrees. Emits `left ++ right` (width `left_width + right_width`);
    /// see [`Join`].
    Join(Join),
    /// A literal row set (`VALUES` / an `INSERT` source). Every inner `Vec` is one
    /// row and all rows have equal width; output width = that width. Each expression
    /// is evaluated with no input row.
    Values { rows: Vec<Vec<EvalExpr>> },
    /// One row of zero columns — a `SELECT` with no `FROM`. The scalar projection
    /// above it supplies the actual output columns.
    SingleRow,
    /// A compound `UNION` / `UNION ALL` / `INTERSECT` / `EXCEPT`; see [`SetOp`].
    SetOp(SetOp),
    /// Scan a materialized CTE / derived table from [`Plan::ctes`]`[id]`. Emits that
    /// CTE's rows; `column_count` is the CTE's output width (carried so the executor
    /// need not consult the CTE plan to size the row).
    CteScan { id: usize, column_count: usize },
    /// Read the current working table inside a recursive-CTE step (the rows produced
    /// by the previous iteration). `column_count` is the recursive CTE's output
    /// width. PROVISIONAL: the recursive-CTE feature refines how the working
    /// table is addressed (this carries only the width the step's row references need).
    RecursiveScan { column_count: usize },
    /// Scan a JSON table-valued function (`json_each` / `json_tree`) — a FROM-clause
    /// row source (json1.html §4.24). `arg` is the JSON document expression and `path`
    /// the optional start-path expression; both are bound against the OUTER row (empty
    /// at top level, or the LEFT row when this is a correlated/LATERAL join right
    /// operand — SQLite table-valued functions are implicitly LATERAL, so the arg may
    /// reference a preceding table's columns). The executor evaluates them per outer row
    /// and generates the rows via [`minisqlite_functions::json_table_rows`]; `kind`
    /// selects the single-level (`Each`) vs recursive (`Tree`) walk. Emits exactly
    /// `column_count` columns — the fixed visible schema
    /// `[key, value, type, atom, id, parent, fullkey, path]` (width
    /// [`minisqlite_functions::JSON_TABLE_COLUMN_COUNT`]) followed by the two hidden input
    /// columns `json`/`root` (json1.html §4.24; excluded from `*`, selectable by name),
    /// with NO trailing rowid — so a `Derived` source of the same width binds against it.
    ///
    /// `emit_json` is `true` iff the statement actually references the hidden `json`
    /// column. It gates the ONLY expensive per-row cost: `json` is the whole document, so
    /// the executor clones it into each row's `json` slot only when `emit_json` (otherwise
    /// that slot is SQL NULL — it is never read). This keeps the common
    /// `SELECT value`/`SELECT *`/`count(*)` scans linear rather than `O(rows·|document|)`.
    /// It is set by a post-binding pass (`compile::select`) from whether the binder
    /// resolved the hidden `json` column, and defaults to `true` (materialize — correct if
    /// conservatively left on).
    TableFunctionScan {
        kind: JsonTableKind,
        arg: EvalExpr,
        path: Option<EvalExpr>,
        column_count: usize,
        emit_json: bool,
    },
    /// Scan a `pragma_*` schema-introspection table-valued function (`pragma_table_info`,
    /// `pragma_index_list`, …) — a FROM-clause row source returning exactly the rows of
    /// the corresponding `PRAGMA` statement (pragma.html §2 "PRAGMA functions"). `kind`
    /// selects which introspection to run; the visible schema is that pragma's fixed
    /// columns, `column_count` wide, with NO trailing rowid — so a `Derived` source of the
    /// same width binds against it.
    ///
    /// `name_arg` is the object-name argument (`pragma_table_info('t')` → `'t'`) and
    /// `schema_arg` the OPTIONAL trailing schema argument (`pragma_table_info('t','main')`
    /// → `'main'`), matching pragma.html: "The PRAGMA argument and schema, if any, are
    /// passed as arguments to the table-valued function, with the schema as an optional,
    /// last argument." Both are bound against the OUTER row (empty at top level, or the
    /// LEFT row when this is a correlated/LATERAL operand — pragma TVFs are implicitly
    /// LATERAL, e.g. `pragma_index_info(il.name)` reads the outer `pragma_index_list`
    /// row). The executor evaluates them PER outer row, coerces each to text, and builds
    /// the rows from the live catalog via [`minisqlite_catalog::pragma_rows`] — the SAME
    /// builder the `PRAGMA` statement form uses, so a TVF and its statement cannot diverge.
    /// A `NULL`/absent object name yields zero rows (the fixed columns still stand), like
    /// the statement form.
    PragmaFunctionScan {
        kind: PragmaFunction,
        name_arg: Option<EvalExpr>,
        schema_arg: Option<EvalExpr>,
        column_count: usize,
    },
    /// Window functions over the input; see [`Window`].
    Window(Window),
    /// `INSERT`; see [`Insert`].
    Insert(Insert),
    /// `UPDATE`; see [`Update`].
    Update(Update),
    /// `DELETE`; see [`Delete`].
    Delete(Delete),
    /// View DML (`INSERT`/`UPDATE`/`DELETE` on a VIEW) redirected through the view's
    /// `INSTEAD OF` trigger(s); see [`InsteadOf`]. A view owns no b-tree, so this makes
    /// NO base-table write — it fires the trigger body per affected row instead.
    InsteadOf(InsteadOf),
    /// `CREATE TABLE`; see [`CreateTablePlan`].
    CreateTable(CreateTablePlan),
    /// `CREATE INDEX`; see [`CreateIndexPlan`].
    CreateIndex(CreateIndexPlan),
}

/// The retention bound a `LIMIT`/`OFFSET` places on the [`Sort`](PlanNode::Sort)
/// directly beneath it: keep at most `offset + limit` rows (a bounded top-k) rather
/// than the whole input. Both are the SAME bound expressions carried on the enclosing
/// [`Limit`](PlanNode::Limit) node — evaluated once, with `OP_MustBeInt` coercion — so
/// the sort and the limit agree on the count.
///
/// The sort uses the bound ONLY as an optimization (it never changes results): the
/// `Limit` node remains the single source of truth for the actual skip/take and for
/// every `LIMIT`/`OFFSET` error/edge case (a `NULL`/non-integral bound errors there; a
/// negative `LIMIT` is unbounded; a negative `OFFSET` is zero). The compiler attaches
/// this ONLY when both expressions are DETERMINISTIC (see
/// [`crate::compile::select::sort_limit_from`]), so the sort's own single evaluation of
/// them matches the `Limit` node's exactly and `retain == offset + limit`; a
/// non-deterministic bound is left off (the sort stays a full stable sort).
#[derive(Debug, Clone)]
pub struct SortLimit {
    /// The `LIMIT` expression (bound with an empty scope, exactly as on the `Limit`
    /// node — no columns are in scope).
    pub limit: EvalExpr,
    /// The `OFFSET` expression, or `None` for no `OFFSET`.
    pub offset: Option<EvalExpr>,
}

/// The join flavor, which decides how unmatched rows are treated (per the
/// convention, the unmatched side's registers are all `Value::Null`).
#[derive(Debug, Clone)]
pub enum JoinType {
    /// Only matched pairs survive.
    Inner,
    /// Every left row survives; unmatched right side is NULL-filled.
    Left,
    /// Every right row survives; unmatched left side is NULL-filled.
    Right,
    /// Every row of both sides survives; the unmatched side is NULL-filled.
    Full,
    /// Cartesian product (no `ON`); every left row paired with every right row.
    Cross,
}

/// How the executor should perform the join. Chosen by the planner from the
/// available access paths and key equalities; the executor runs the named strategy
/// without re-deciding.
#[derive(Debug, Clone)]
pub enum JoinStrategy {
    /// Nested-loop join: for each left row, scan the right subtree. The general
    /// fallback (and the only shape for a non-equi `ON` or a `CROSS` join).
    NestedLoop,
    /// Hash join on equality keys: build a hash table on one side keyed by
    /// `right_keys`, probe it with `left_keys`. `key_collations` gives the collation
    /// for each key pair so text keys hash/compare consistently. All three vectors
    /// are the same length (one entry per equijoin key).
    Hash { left_keys: Vec<EvalExpr>, right_keys: Vec<EvalExpr>, key_collations: Vec<Collation> },
    /// Index nested-loop: the right subtree is an [`IndexScan`]/[`RowidScan`] whose
    /// key expressions reference the current left row, so each left row drives a seek
    /// rather than a full right scan.
    IndexNestedLoop,
}

/// A join operator. `on` is the join predicate (`None` for a `CROSS`/comma join).
/// Widths are carried so the executor lays the combined row out as data.
#[derive(Debug, Clone)]
pub struct Join {
    /// Left input subtree; its rows occupy registers `[0, left_width)`.
    pub left: Box<PlanNode>,
    /// Width of a left row (register offset where the right row begins).
    pub left_width: usize,
    /// Right input subtree; its rows occupy registers `[left_width, left_width+right_width)`.
    pub right: Box<PlanNode>,
    /// Width of a right row.
    pub right_width: usize,
    /// Inner / outer / cross flavor.
    pub join_type: JoinType,
    /// The `ON` predicate, bound against the combined `left ++ right` row. `None` for
    /// a `CROSS`/comma join with no condition.
    pub on: Option<EvalExpr>,
    /// The physical strategy the executor runs.
    pub strategy: JoinStrategy,
}

/// Grouping and aggregation.
///
/// Emits `[group_keys…, agg_results…]`: the `group_by` key values first (in order),
/// then one result per [`AggregateCall`]. `DISTINCT`, `FILTER`, and aggregate
/// `ORDER BY` are carried on each `AggregateCall` and applied by the operator around
/// the accumulator. With no `group_by` over an empty input the operator still emits
/// exactly one row (the aggregates over zero rows).
///
/// When [`minmax_bare`](Aggregate::minmax_bare) or
/// [`bare_arbitrary`](Aggregate::bare_arbitrary) is set the operator additionally
/// appends the captured bare columns, emitting `[group_keys…, agg_results…, captured…]`.
#[derive(Debug, Clone)]
pub struct Aggregate {
    /// The rows to aggregate.
    pub input: Box<PlanNode>,
    /// Grouping key expressions, evaluated against an input row. Empty = a single
    /// implicit group over the whole input.
    pub group_by: Vec<EvalExpr>,
    /// Collation for each `group_by` key (same length as `group_by`), so text keys
    /// group under the right collating sequence.
    pub group_collations: Vec<Collation>,
    /// The aggregate invocations, in output order after the group keys.
    pub aggregates: Vec<AggregateCall>,
    /// `HAVING` predicate, bound against the emitted `[group_keys…, agg_results…]`
    /// row (post-aggregation). `None` = no `HAVING`.
    pub having: Option<EvalExpr>,
    /// The single-min/max "bare columns" special case (lang_select.html §2.5), when
    /// it applies; `None` for every ordinary aggregate query. See [`MinMaxBare`].
    pub minmax_bare: Option<MinMaxBare>,
    /// The GENERAL bare-column case (lang_select.html §2.5): input registers to capture
    /// for bare columns in an aggregate query that is NOT the single-min/max special case
    /// (no min/max, or two-or-more). SQLite lets a bare column take its value from an
    /// arbitrary input row of the group; the operator captures these registers from the
    /// FIRST row of each group (deterministic, and identical to any row for a
    /// functionally-dependent bare column) and appends them, in this order, after the
    /// aggregate results — the same output slots the projection binds a bare column to.
    /// `None`/empty for a query with no bare columns, and always `None` when `minmax_bare`
    /// is `Some` (that case captures from the extremum row instead). Mutually exclusive
    /// with `minmax_bare`.
    pub bare_arbitrary: Option<Vec<usize>>,
}

/// The single-`min()`/`max()` "bare columns" special case (lang_select.html §2.5).
///
/// SQLite documents that when a query has exactly one `min()` or `max()` aggregate,
/// every *bare* result column (a column neither inside an aggregate nor in `GROUP BY`)
/// takes its value from the input row that produced that extremum. This engine applies
/// the case whenever the query has EXACTLY ONE aggregate-form `min()`/`max()` call —
/// regardless of how many OTHER aggregates accompany it (§2.5 only bars the case when
/// there are *two or more* min/max aggregates). The captured columns follow all
/// aggregate results, so they land at the known offset `num_keys + aggregates.len()`,
/// computable while the projection is still being bound (the compile-side pre-scan
/// counts the aggregates the binder will collect).
///
/// The [`crate::plan::PlanNode::Aggregate`] operator captures `captured_regs` from the
/// extremum row and appends them after the aggregate results, so the emitted row is
/// `[group_keys.., agg_results.., captured..]` and a bare column at slot `k` reads
/// `Column(num_keys + aggregates.len() + k)`.
#[derive(Debug, Clone)]
pub struct MinMaxBare {
    /// Index into [`Aggregate::aggregates`] of the min/max aggregate whose extremum
    /// row supplies the bare columns. `0` when the min/max is the only (or first)
    /// aggregate, but non-zero when other aggregates precede it (e.g. `count(*),
    /// max(v)`); the operator watches this slot rather than assuming index 0.
    pub agg_index: usize,
    /// `true` for `max()`, `false` for `min()`: the direction the extremum advances.
    /// Mirrors the built-in min/max (`minisqlite-functions` `agg/minmax.rs`), which
    /// replaces its best only on a strict win, so ties keep the first-seen row.
    pub is_max: bool,
    /// Input-row registers to capture from the extremum row, in output-slot order.
    /// The operator appends these values (or NULLs for a group with no non-NULL
    /// extremum) after the aggregate results.
    pub captured_regs: Vec<usize>,
}

/// The MIN/MAX index/rowid seek fast path (optoverview.html §13, "The MIN/MAX
/// Optimization"): compute a single `MIN(col)`/`MAX(col)` over an ENTIRE table with one
/// O(log n) b-tree seek to the extremum entry instead of an O(n) full-scan aggregate.
///
/// The planner emits this in place of the ordinary [`Aggregate`] ONLY when the whole
/// aggregate is provably a single bare-column MIN/MAX with a BINARY comparison served by
/// the rowid or a usable ascending index (see `compile::minmax_index`); every case it
/// cannot prove byte-identical stays on the full-scan aggregate. The executor emits
/// EXACTLY ONE width-1 row `[value]`, so the projection above it (`MAX(x)+1`, column
/// naming) binds against `Column(0)` exactly as it did for the single-aggregate
/// `Aggregate` output — only the VALUE computation changes.
///
/// # Extremum + NULL contract (the correctness core)
///
/// MIN/MAX ignore NULLs and return NULL iff there is no non-NULL value
/// (`lang_aggfunc.html`). In an ASCENDING BINARY b-tree NULLs sort FIRST, so:
/// * **MAX**: the extremum is the LAST entry (largest key); its value is non-NULL unless
///   the table is empty or every value is NULL — both yield NULL.
/// * **MIN**: the extremum is the FIRST NON-NULL entry — walk from the smallest key and
///   skip leading NULL-keyed entries; empty or all-NULL yields NULL.
///
/// The value is byte-identical to the scan-path aggregate because ties (equal-comparing
/// values) are only representation-ambiguous across storage classes for a typeless (BLOB
/// affinity) column, which the planner excludes from the [`MinMaxSource::Index`] path; the
/// rowid key is unique, so [`MinMaxSource::Rowid`] has no ties.
#[derive(Debug, Clone)]
pub struct MinMaxSeek {
    /// The base table whose b-tree (or whose index b-tree) is seeked.
    pub table: String,
    /// The namespace the table (and its index) live in (see [`SeqScan::db`]).
    pub db: DbIndex,
    /// `true` for `MAX` (seek the largest key), `false` for `MIN` (the smallest non-NULL).
    pub is_max: bool,
    /// Which b-tree carries the ordered extremum.
    pub source: MinMaxSource,
}

/// The b-tree a [`MinMaxSeek`] walks to reach the extremum.
#[derive(Debug, Clone)]
pub enum MinMaxSource {
    /// The table's own rowid b-tree — the aggregate argument is the rowid / an
    /// `INTEGER PRIMARY KEY` alias / `rowid`/`_rowid_`/`oid`. MAX = the last (largest)
    /// rowid, MIN = the first (smallest); the value is the rowid as `Value::Integer`, and
    /// an empty table yields NULL. Rowids are unique and never NULL, so there are no ties.
    Rowid,
    /// A named ascending, non-partial index whose LEFT-MOST key column is the aggregate
    /// argument and whose declared collation for that column is BINARY (the physical
    /// b-tree order). The extremum value is decoded from the index entry's leftmost key
    /// column.
    Index {
        /// The index's name (its root page is resolved through the catalog at run time).
        index: String,
    },
}

/// Which compound set operator combines the two inputs.
#[derive(Debug, Clone)]
pub enum SetOpKind {
    /// `UNION ALL` — concatenate, keep duplicates.
    UnionAll,
    /// `UNION` — concatenate then remove duplicate rows.
    Union,
    /// `INTERSECT` — distinct rows present in both inputs.
    Intersect,
    /// `EXCEPT` — distinct rows of the left not present in the right.
    Except,
}

/// A compound select. Both inputs must be the same width; the output is that width,
/// and result column names come from the left input (per the convention).
#[derive(Debug, Clone)]
pub struct SetOp {
    /// The set operator to apply.
    pub op: SetOpKind,
    /// Left input subtree.
    pub left: Box<PlanNode>,
    /// Right input subtree (same width as the left).
    pub right: Box<PlanNode>,
    /// Per-output-column collation used for the duplicate/`INTERSECT`/`EXCEPT`
    /// comparisons (same length as the output width).
    pub column_collations: Vec<Collation>,
}

/// A materialized CTE or derived table, referenced from [`PlanNode::CteScan`] by its
/// index in [`Plan::ctes`].
#[derive(Debug, Clone)]
pub enum CtePlan {
    /// A plain (non-recursive) materialized CTE / derived table: run `body` once and
    /// scan the buffered rows. `column_count` is its output width.
    Materialized { name: String, column_count: usize, body: PlanNode },
    /// A recursive CTE: run `seed`, then repeatedly run `step` (which reads the
    /// previous iteration's rows via [`PlanNode::RecursiveScan`]) until it yields no
    /// new rows, unioning results. `union_all` selects `UNION ALL` (keep duplicates)
    /// vs `UNION` (dedup). `column_count` is the CTE's output width.
    ///
    /// A body `LIMIT`/`OFFSET` is NOT carried here: the compiler lowers it to a
    /// [`CtePlan::Materialized`] wrapping a [`PlanNode::Limit`] over this fixpoint's
    /// [`PlanNode::CteScan`]. Because the fixpoint streams lazily, that wrapper's `LIMIT`
    /// also bounds an otherwise-infinite body-`LIMIT` recursion by stopping its pulls
    /// (spec `lang_with.html` §3).
    Recursive {
        name: String,
        column_count: usize,
        seed: PlanNode,
        step: PlanNode,
        union_all: bool,
    },
}

/// An expression subquery's plan, referenced from an `EvalExpr` subquery id (its
/// index in [`Plan::subqueries`]).
#[derive(Debug, Clone)]
pub struct SubPlan {
    /// The subquery's operator tree.
    pub plan: PlanNode,
    /// `true` if the subquery references columns of the outer row (a correlated
    /// subquery): its `EvalExpr::Column(i)` for `i < outer_width` reads the outer row
    /// passed through the `EvalContext` callback, so the executor must re-run it per
    /// outer row rather than once.
    pub correlated: bool,
    /// The COMPLETE, deduplicated, sorted set of OUTER registers (each `< outer_width`)
    /// this subplan's result depends on, transitively including any nested subqueries
    /// that reach through this one to an enclosing row. EMPTY for an uncorrelated
    /// subplan (`correlated == false`).
    ///
    /// A memoizing evaluator builds a per-outer-row cache key from the VALUES of these
    /// registers — `outer_row[r]` for each `r` in `correlated_cols` — so two outer rows
    /// with equal values at exactly these registers share the subplan's result. That is
    /// why completeness is a CORRECTNESS invariant, not an optimization detail: a missed
    /// register would let two outer rows that actually differ (in that register) collide
    /// on one cache entry and return the wrong answer. Collected at the SAME
    /// successful-parent-resolve site as `correlated` (the binder's `resolve_via_parent`),
    /// so it is exactly as complete and transitive as the `correlated` flag itself.
    pub correlated_cols: Vec<usize>,
    /// `true` only when the subplan is KNOWN to be deterministic: its subtree contains no
    /// non-deterministic function (a `ScalarFunction` whose `deterministic()` is `false`,
    /// e.g. `random()`) and no `EvalExpr::Now(_)`
    /// (`CURRENT_DATE`/`CURRENT_TIME`/`CURRENT_TIMESTAMP`), transitively through nested
    /// subqueries, AND it reaches no un-analyzed materialized body. A memoizing evaluator
    /// must only reuse a cached result when this holds: a subplan that can yield a different
    /// value on a second evaluation with the same outer input is unsafe to memoize.
    ///
    /// Conservative — anything the analysis cannot PROVE deterministic is reported `false`,
    /// which only forgoes the optimization (the evaluator falls back to re-running), never a
    /// wrong answer. In particular a subplan whose plan reaches a CTE / derived-table scan
    /// (`FROM (SELECT …)`, or a reference to an enclosing `WITH`) is reported `false`
    /// unconditionally: that body is compiled through a path that does not thread the
    /// determinism analysis, so a `random()` inside it cannot be ruled out — and since the
    /// body is re-materialized per outer-row re-open, such a `random()` would in fact make
    /// the subplan non-deterministic. The flag is thus SUFFICIENT for memoization safety
    /// (with respect to functions/clock/materialized bodies) but not a precise "contains no
    /// impurity" oracle; a determined caller wanting to memoize the deterministic-CTE case
    /// would need to analyze the CTE bodies too.
    pub deterministic: bool,
    /// The register width this subplan was compiled against — `parent.total_width()`, the
    /// size of the outer row prepended to each of the subplan's own rows — so every entry
    /// of `correlated_cols` is `< outer_width`. `0` for an uncorrelated subplan. Metadata
    /// a memoizing evaluator can use to size / validate the outer-row slice it keys on.
    pub outer_width: usize,
}

/// The conflict-resolution algorithm for a constraint violation during DML
/// (`INSERT OR …` / `UPDATE OR …`, or a table/column `ON CONFLICT`).
#[derive(Debug, Clone)]
pub enum OnConflict {
    /// Abort the statement and undo its changes (the default), erroring.
    Abort,
    /// Abort the whole transaction, erroring.
    Rollback,
    /// Stop at the failing row but keep prior changes of the statement, erroring.
    Fail,
    /// Skip the offending row and continue without error.
    Ignore,
    /// Delete the conflicting existing row(s), then proceed with the write.
    Replace,
}

/// A compiled UPSERT: the `ON CONFLICT ...` clauses of an INSERT, matched per row
/// (`lang_upsert.html`). Carried on [`Insert::upsert`] (`None` for a plain INSERT). The
/// executor, per inserted row, detects a uniqueness conflict; on one it runs the first
/// matching clause (an in-place UPDATE of the existing row, or a no-op) INSTEAD of the
/// insert; with no conflict it inserts normally.
#[derive(Debug, Clone)]
pub struct UpsertPlan {
    /// The chained clauses, in written order. The first clause whose `target` matches
    /// the violated constraint runs (a target-omitted clause matches any conflict).
    pub clauses: Vec<UpsertClause>,
}

/// One resolved `ON CONFLICT [target] DO ...` clause.
#[derive(Debug, Clone)]
pub struct UpsertClause {
    /// The resolved conflict target — which uniqueness constraint this clause handles.
    pub target: ConflictTarget,
    pub action: UpsertActionPlan,
}

/// A resolved `ON CONFLICT` conflict target (`lang_upsert.html` §2): the uniqueness
/// constraint a clause matches, resolved at plan time against the schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictTarget {
    /// Target omitted (`ON CONFLICT DO ...`, the last clause only): matches ANY uniqueness
    /// conflict not already captured by a prior clause.
    Any,
    /// A column-set target: a PRIMARY KEY / rowid-alias column set, or a UNIQUE index's
    /// column set, as schema column indices. Two column targets match order-insensitively
    /// (the column SET, not its order) — the executor compares via `same_column_set`.
    Columns(Vec<usize>),
    /// A specific UNIQUE index named by its b-tree ROOT PAGE — the same identity the
    /// executor's `IndexPlan` carries — used for the two targets a plain column SET cannot
    /// describe:
    /// * an EXPRESSION (or mixed name+expression) target (`ON CONFLICT(a+b)` matching
    ///   `CREATE UNIQUE INDEX … ON t(a+b)`), pinned to the one index whose key it
    ///   structurally equals; and
    /// * a PARTIAL-index target (`ON CONFLICT(<cols-or-exprs>) WHERE <pred>` matching
    ///   `CREATE UNIQUE INDEX … WHERE <pred>`), pinned to the one partial index whose key AND
    ///   `WHERE` predicate it structurally equals — a partial index (column- or
    ///   expression-based) is named by its root, not a plain column set, because the executor
    ///   detects its conflict by index root (only that index's admitted rows are in it).
    ///
    /// (`u32` is the pager's `PageId`; the plan crate does not depend on the pager, so the raw
    /// page number is stored directly.)
    ExprIndex(u32),
}

/// What a matched UPSERT clause does on conflict.
#[derive(Debug, Clone)]
pub enum UpsertActionPlan {
    /// `DO NOTHING`: skip the offending row (no insert, no update, no error).
    Nothing,
    /// `DO UPDATE SET <assignments> [WHERE <predicate>]`, an in-place UPDATE of the
    /// EXISTING conflicting row. Both the assignment values and the predicate are bound
    /// over the combined row `existing(W) ++ excluded(W)` (`W` = the target table's
    /// base-row width `N+1`): a bare / `table.`-qualified column and `rowid` read the
    /// EXISTING row `[0, W)`; an `excluded.col` reads the would-be-inserted row `[W, 2W)`.
    /// `assignments` are `(schema column index, value expr)`; `predicate` (the `WHERE`),
    /// when present, gates the update — a NULL/false result makes it a no-op.
    Update { assignments: Vec<(usize, EvalExpr)>, predicate: Option<EvalExpr> },
}

/// A compiled trigger: the executable form of one `CREATE TRIGGER`, ready for the
/// firing executor to run when a write touches the trigger's target table. It is
/// produced by the trigger-compile step from the catalog's stored
/// `CREATE TRIGGER` text and carried on the [`Insert`] / [`Update`] / [`Delete`] node
/// whose table the trigger fires on (their `triggers` field). This scaffold only
/// defines the shape and its register contract; nothing populates or fires it yet.
///
/// # NEW / OLD register convention (the binding contract)
///
/// A trigger's action statements and its `WHEN` condition reference the pseudo-rows
/// `NEW` and `OLD` (`NEW.col`, `OLD.col`, `NEW.rowid`, `OLD.rowid`). Both the
/// trigger-compile step — which binds those references into
/// [`EvalExpr::Column`](minisqlite_expr::EvalExpr::Column) — and the firing executor —
/// which supplies the register values — MUST obey the layout below. It is the same
/// correlated-outer-register scheme the module doc describes for a correlated subplan,
/// with `OLD ++ NEW` playing the role of the outer row.
///
/// Let `C` be the target table's column count and `W = C + 1` its base-row width (the
/// `[cols…, rowid]` width a base scan emits). When the executor runs an action or
/// evaluates `when`, it places the OLD row in the leading registers `[0, W)` and the
/// NEW row in `[W, 2W)`, and the action's own operators bind their columns starting at
/// register `2W`. A reference therefore resolves to:
///
/// * `OLD.col_k` → `Column(k)` for `k` in `[0, C)`
/// * `OLD.rowid` → `Column(C)`
/// * `NEW.col_k` → `Column(W + k)` for `k` in `[0, C)`
/// * `NEW.rowid` → `Column(W + C)`
///
/// By event, one half is absent and filled with `Value::Null`: an `INSERT` trigger has
/// no OLD row (registers `[0, W)` are all NULL), a `DELETE` trigger has no NEW row
/// (`[W, 2W)` are all NULL), and an `UPDATE` trigger populates both halves.
#[derive(Debug, Clone)]
pub struct TriggerProgram {
    /// The trigger's name — for error messages and to identify it deterministically.
    pub name: String,
    /// When it fires relative to its event: `BEFORE` / `AFTER` / `INSTEAD OF`.
    pub timing: minisqlite_sql::TriggerTiming,
    /// The `WHEN` condition, bound over the `OLD ++ NEW` leading registers per the
    /// convention above; `None` when the trigger has no `WHEN` (it fires unconditionally).
    pub when: Option<EvalExpr>,
    /// The compiled action statements (`INSERT` / `UPDATE` / `DELETE` / `SELECT`),
    /// each a full [`Plan`], run in order when the trigger fires.
    pub actions: Vec<Plan>,
}

/// A compiled CHECK constraint carried on an [`Insert`]/[`Update`] node: a bound
/// predicate over the new row `[c0..c_{N-1}, rowid]` (column `i` binds to register `i`,
/// and an INTEGER PRIMARY KEY alias column binds to the trailing rowid register `N` — the
/// same width-`N+1` layout the executor's logical row uses) plus the detail string for the
/// error message `CHECK constraint failed: <detail>`. The executor evaluates `expr` against
/// each new row and, per lang_createtable.html §3.7, treats an integer-0 / real-0.0
/// result (after CAST-to-NUMERIC) as a violation while NULL or any nonzero value passes.
#[derive(Debug, Clone)]
pub struct CheckConstraint {
    pub expr: EvalExpr,
    pub detail: String,
}

/// `INSERT`. The `source` subtree produces the rows to insert (a [`PlanNode::Values`]
/// or a `SELECT` plan); each source row is mapped onto the target columns, affinity
/// is applied, and the row is written under `on_conflict`.
#[derive(Debug, Clone)]
pub struct Insert {
    /// Target table name.
    pub table: String,
    /// Which database namespace the target table lives in (see
    /// [`SeqScan::db`](crate::access::SeqScan::db)). The write applies to this store; a
    /// cross-namespace source (`INSERT INTO main.t SELECT … FROM temp.s`) is read in the
    /// buffered source phase, but the write only ever touches this namespace.
    pub db: DbIndex,
    /// The table's column count `N` (the inserted row shape is `[cols…, rowid]`,
    /// width `N+1`).
    pub column_count: usize,
    /// Target column indices in the order the source supplies them; `None` means all
    /// columns in `CREATE TABLE` order. Entries are real column indices in `[0, N)` with
    /// ONE exception: the SENTINEL `N` (== `column_count`) at the position that named
    /// `rowid`/`_rowid_`/`oid` on a plain rowid table — that entry is NOT a stored column
    /// but the explicit-rowid channel (see `rowid_source`). A consumer that indexes a
    /// per-column structure (`td.columns`, `column_affinities`) by a `columns` entry MUST
    /// skip the sentinel slot.
    pub columns: Option<Vec<usize>>,
    /// The rows to insert.
    pub source: Box<PlanNode>,
    /// Width of a `source` row.
    pub source_width: usize,
    /// Per table-column affinity, applied to each inserted value as it is stored
    /// (length `N`).
    pub column_affinities: Vec<Affinity>,
    /// Conflict-resolution algorithm for constraint violations.
    pub on_conflict: OnConflict,
    /// `RETURNING` expressions, evaluated against the inserted row layout
    /// `[cols…, rowid]`. Empty when there is no `RETURNING` clause.
    pub returning: Vec<EvalExpr>,
    /// Triggers that fire for this `INSERT`, in firing order (empty when the target has
    /// none). Populated by the trigger-compile pass (`compile::trigger::compile_triggers`);
    /// the firing executor runs each per [`TriggerProgram`]'s NEW/OLD register convention
    /// (for an INSERT the OLD half is all-NULL).
    pub triggers: Vec<TriggerProgram>,
    /// The target table's CHECK constraints, bound against the inserted row layout
    /// `[c0..c_{N-1}, rowid]` (empty when the table has none). Populated by
    /// `compile::check::compile_checks`; the executor evaluates each per inserted row and
    /// aborts on a violation (0/0.0 after CAST-to-NUMERIC; NULL/nonzero passes). See
    /// [`CheckConstraint`].
    pub checks: Vec<CheckConstraint>,
    /// For a PLAIN rowid table (no INTEGER PRIMARY KEY alias) whose INSERT target list
    /// named `rowid`/`_rowid_`/`oid`: the index (into each source row, equivalently into
    /// `columns`) that supplies the explicit integer rowid. `None` means the rowid is
    /// auto-assigned, or (for an INTEGER PRIMARY KEY table) comes from the alias column
    /// named in `columns`. The `columns` entry at this position is the SENTINEL
    /// `column_count` (== N) — NOT a stored column. The executor reads
    /// `src_row[rowid_source]` (coerced via MustBeInt) as the rowid and skips storing the
    /// sentinel position as a column.
    pub rowid_source: Option<usize>,
    /// The compiled `ON CONFLICT ...` UPSERT clauses, or `None` for a plain INSERT. When
    /// `Some`, the executor makes a per-row upsert decision (insert / in-place update /
    /// skip) instead of the plain `on_conflict` handling. See [`UpsertPlan`].
    pub upsert: Option<UpsertPlan>,
    /// Compiled key expressions for the target table's EXPRESSION indexes. One entry per
    /// index OF THE TARGET TABLE that has >=1 EXPRESSION key column; `String` is the index
    /// name, the `Vec<Option<EvalExpr>>` is aligned with that index's `columns`/`key_exprs`
    /// (`Some` = the compiled key expression for an expression key column, `None` = an
    /// ordinary named column). Indexes whose key is entirely ordinary columns are OMITTED
    /// (so the common case is an empty vec). Populated by the DML compilers; consumed by the
    /// executor's index maintenance.
    pub index_key_exprs: Vec<(String, Vec<Option<EvalExpr>>)>,
    /// Bound WHERE predicates for the target table's PARTIAL indexes. One entry per PARTIAL
    /// index of the target table; `String` is the index name, the `EvalExpr` is the predicate
    /// bound against the row layout `[c0..c_{N-1}, rowid]`. NON-partial indexes are OMITTED
    /// (so the common case is an empty vec). Populated by the DML compilers
    /// (`compile::index_expr::compile_table_index_partial_predicates`); the executor evaluates
    /// each per row and includes/checks/removes that index's entry ONLY when the predicate is
    /// TRUE (`partialindex.html` §2) — so a UNIQUE partial index is enforced only across the
    /// rows actually in it, not over every row.
    pub index_partial_predicates: Vec<(String, EvalExpr)>,
}

/// `UPDATE`. The `scan` subtree produces the rows to update (each `[cols…, rowid]`);
/// for each, `assignments` compute new column values, affinity is applied, and the
/// row is rewritten under `on_conflict`.
#[derive(Debug, Clone)]
pub struct Update {
    /// Target table name.
    pub table: String,
    /// Which database namespace the target table lives in (see
    /// [`SeqScan::db`](crate::access::SeqScan::db)).
    pub db: DbIndex,
    /// The table's column count `N` (the scanned/rewritten row is `[cols…, rowid]`,
    /// width `N+1`).
    pub column_count: usize,
    /// `(column index, value expr)` pairs; each value expr is evaluated against the
    /// scanned row `[cols…, rowid]`.
    pub assignments: Vec<(usize, EvalExpr)>,
    /// Rows to update: a scan (+ filter) producing `[cols…, rowid]`.
    pub scan: Box<PlanNode>,
    /// Per table-column affinity, applied to each assigned value (length `N`).
    pub column_affinities: Vec<Affinity>,
    /// Conflict-resolution algorithm for constraint violations.
    pub on_conflict: OnConflict,
    /// `RETURNING` expressions, evaluated against the updated row `[cols…, rowid]`.
    /// Empty when there is no `RETURNING` clause.
    pub returning: Vec<EvalExpr>,
    /// Triggers that fire for this `UPDATE`, in firing order (empty when the target has
    /// none, or none survives the assigned columns' `UPDATE OF` filter). Populated by the
    /// trigger-compile pass (`compile::trigger::compile_triggers`); the firing executor
    /// runs each per [`TriggerProgram`]'s NEW/OLD register convention (an UPDATE populates
    /// both the OLD and NEW halves).
    pub triggers: Vec<TriggerProgram>,
    /// The target table's CHECK constraints, bound against the row layout
    /// `[c0..c_{N-1}, rowid]` (empty when the table has none). Populated by
    /// `compile::check::compile_checks`; the executor evaluates each against the
    /// POST-assignment new row and aborts on a violation (0/0.0 after CAST-to-NUMERIC;
    /// NULL/nonzero passes). See [`CheckConstraint`].
    pub checks: Vec<CheckConstraint>,
    /// Compiled key expressions for the target table's EXPRESSION indexes. One entry per
    /// index OF THE TARGET TABLE that has >=1 EXPRESSION key column; `String` is the index
    /// name, the `Vec<Option<EvalExpr>>` is aligned with that index's `columns`/`key_exprs`
    /// (`Some` = the compiled key expression for an expression key column, `None` = an
    /// ordinary named column). Indexes whose key is entirely ordinary columns are OMITTED
    /// (so the common case is an empty vec). Populated by the DML compilers; consumed by the
    /// executor's index maintenance.
    pub index_key_exprs: Vec<(String, Vec<Option<EvalExpr>>)>,
    /// Bound WHERE predicates for the target table's PARTIAL indexes. One entry per PARTIAL
    /// index of the target table; `String` is the index name, the `EvalExpr` is the predicate
    /// bound against the row layout `[c0..c_{N-1}, rowid]`. NON-partial indexes are OMITTED
    /// (so the common case is an empty vec). Populated by the DML compilers
    /// (`compile::index_expr::compile_table_index_partial_predicates`); the executor evaluates
    /// each per row and includes/checks/removes that index's entry ONLY when the predicate is
    /// TRUE (`partialindex.html` §2) — so a UNIQUE partial index is enforced only across the
    /// rows actually in it, not over every row.
    pub index_partial_predicates: Vec<(String, EvalExpr)>,
}

/// `DELETE`. The `scan` subtree produces the rows to delete (each `[cols…, rowid]`);
/// each is removed by its rowid.
#[derive(Debug, Clone)]
pub struct Delete {
    /// Target table name.
    pub table: String,
    /// Which database namespace the target table lives in (see
    /// [`SeqScan::db`](crate::access::SeqScan::db)).
    pub db: DbIndex,
    /// The table's column count `N` (the scanned row is `[cols…, rowid]`, width
    /// `N+1`).
    pub column_count: usize,
    /// Rows to delete: a scan (+ filter) producing `[cols…, rowid]`.
    pub scan: Box<PlanNode>,
    /// `RETURNING` expressions, evaluated against the deleted row `[cols…, rowid]`.
    /// Empty when there is no `RETURNING` clause.
    pub returning: Vec<EvalExpr>,
    /// Triggers that fire for this `DELETE`, in firing order (empty when the target has
    /// none). Populated by the trigger-compile pass (`compile::trigger::compile_triggers`);
    /// the firing executor runs each per [`TriggerProgram`]'s NEW/OLD register convention
    /// (for a DELETE the NEW half is all-NULL).
    pub triggers: Vec<TriggerProgram>,
    /// Compiled key expressions for the target table's EXPRESSION indexes. One entry per
    /// index OF THE TARGET TABLE that has >=1 EXPRESSION key column; `String` is the index
    /// name, the `Vec<Option<EvalExpr>>` is aligned with that index's `columns`/`key_exprs`
    /// (`Some` = the compiled key expression for an expression key column, `None` = an
    /// ordinary named column). Indexes whose key is entirely ordinary columns are OMITTED
    /// (so the common case is an empty vec). Populated by the DML compilers; consumed by the
    /// executor's index maintenance.
    pub index_key_exprs: Vec<(String, Vec<Option<EvalExpr>>)>,
    /// Bound WHERE predicates for the target table's PARTIAL indexes. One entry per PARTIAL
    /// index of the target table; `String` is the index name, the `EvalExpr` is the predicate
    /// bound against the row layout `[c0..c_{N-1}, rowid]`. NON-partial indexes are OMITTED
    /// (so the common case is an empty vec). Populated by the DML compilers
    /// (`compile::index_expr::compile_table_index_partial_predicates`); the executor evaluates
    /// each per row and includes/checks/removes that index's entry ONLY when the predicate is
    /// TRUE (`partialindex.html` §2) — so a UNIQUE partial index is enforced only across the
    /// rows actually in it, not over every row.
    pub index_partial_predicates: Vec<(String, EvalExpr)>,
}

/// Which DML event a view's `INSTEAD OF` trigger(s) redirect (see [`InsteadOf`]). A
/// view is updatable for a given event ONLY when it has a matching `INSTEAD OF <event>`
/// trigger (`lang_createtrigger.html` §3); this records which one the node handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsteadOfEvent {
    Insert,
    Update,
    Delete,
}

/// A view DML (`INSERT`/`UPDATE`/`DELETE` on a VIEW) redirected through the view's
/// `INSTEAD OF` trigger(s) (`lang_createtrigger.html` §3). A view owns no b-tree, so
/// there is NO base-table write: for each affected view row the executor fires every
/// matching `INSTEAD OF` program's body FOR EACH ROW, binding OLD/NEW.
///
/// # Frame contract (why the executor needs no event-specific shaping)
///
/// `frame_source` yields exactly ONE `OLD ++ NEW` frame per affected row, already the
/// width `2W` (`W = column_count + 1`, `column_count` = the view's column count `C`).
/// This is the SAME `OLD ++ NEW` register layout a base-table [`TriggerProgram`] binds
/// against (see its doc): `OLD.col_k → Column(k)`, `NEW.col_k → Column(W+k)`, and each
/// action's own operators at register `2W`. A view has no rowid, so both rowid slots
/// (register `C` and `2C+1`) are NULL placeholders. By event one half is all-NULL —
/// INSERT has no OLD, DELETE has no NEW — and UPDATE populates both; the planner builds
/// that shape into `frame_source`, so the executor simply drives each frame through the
/// programs (no per-event branching at run time).
#[derive(Debug, Clone)]
pub struct InsteadOf {
    /// The view's name (creation-case spelling), for error messages / debugging.
    pub view: String,
    /// Which DML event is being redirected (INSERT / UPDATE / DELETE).
    pub event: InsteadOfEvent,
    /// The view's column count `C`; each `frame_source` row is `2*(C+1)` wide.
    pub column_count: usize,
    /// Produces one `OLD ++ NEW` frame (width `2*(C+1)`) per affected view row.
    pub frame_source: Box<PlanNode>,
    /// The matching `INSTEAD OF` programs, in catalog (firing) order; ALL fire per frame
    /// (each gated by its own `WHEN`). Empty is never emitted — the compiler keeps the
    /// "cannot modify … because it is a view" error when no `INSTEAD OF` trigger matches.
    pub programs: Vec<TriggerProgram>,
    /// Bound `RETURNING` output expressions (empty when the view DML has no `RETURNING`).
    /// Evaluated once per affected view row against that row's view columns `[0, C)` — the
    /// NEW half of the frame for INSERT/UPDATE, the OLD half for DELETE — producing one
    /// result row each. Per `lang_returning.html` these are the values "as seen by the
    /// top-level … statement", i.e. what the frame holds BEFORE the fired body runs, so the
    /// executor evaluates them before firing (`ops::instead_of`). `EvalExpr::Column(i)`
    /// indexes view column `i`; the executor supplies the C-wide half as both the row and a
    /// correlated RETURNING subquery's outer.
    pub returning: Vec<EvalExpr>,
}

/// One column of a `CREATE TABLE`: its name and the raw declared-type text (from
/// which affinity is derived). Kept minimal — richer constraint facts live in the
/// catalog's `TableDef` once the table is registered.
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub name: String,
    /// The declared type text as written (`None` for a typeless column).
    pub declared_type: Option<String>,
}

/// `CREATE TABLE`: register a new table's schema.
#[derive(Debug, Clone)]
pub struct CreateTablePlan {
    pub name: String,
    /// The columns in declaration order.
    pub columns: Vec<ColumnSpec>,
    /// `IF NOT EXISTS` — a no-op (not an error) when the table already exists.
    pub if_not_exists: bool,
}

/// `CREATE INDEX`: register a new index on an existing table.
#[derive(Debug, Clone)]
pub struct CreateIndexPlan {
    pub name: String,
    /// The table the index is defined on.
    pub table: String,
    /// The indexed column names, in index key order.
    pub columns: Vec<String>,
    /// `UNIQUE` index.
    pub unique: bool,
    /// `IF NOT EXISTS` — a no-op (not an error) when the index already exists.
    pub if_not_exists: bool,
}

/// Window functions over the input rows.
///
/// Each [`WindowFunc`] appends one column to the input row, so the output is the input
/// row followed by one value per function (width = input width + `functions.len()`).
/// The per-function partition, ordering, and frame live on each [`WindowFunc`].
#[derive(Debug, Clone)]
pub struct Window {
    /// The rows the window functions run over.
    pub input: Box<PlanNode>,
    /// The window functions to evaluate; each appends one output column.
    pub functions: Vec<WindowFunc>,
}

/// One window-function invocation: which function ([`WindowFuncKind`]), the partition
/// and ordering it runs under, and the frame it evaluates over.
///
/// `partition_by` splits the input into independent partitions; `order_by` orders rows
/// within a partition; `frame` bounds the sub-range each row's function sees
/// ([`WindowFrame::default_frame`] when the window carries no explicit `frame-spec`).
/// The ranking functions and `lag`/`lead` are defined by row position and ignore the
/// frame, but it is still carried so the shape is uniform across every kind.
#[derive(Debug, Clone)]
pub struct WindowFunc {
    /// Which window function this invocation computes.
    pub kind: WindowFuncKind,
    /// `PARTITION BY` key expressions (empty = one partition over the whole input).
    pub partition_by: Vec<EvalExpr>,
    /// The §7.1 collating sequence for each `PARTITION BY` key (aligned with
    /// `partition_by`, BINARY when absent). `PARTITION BY` groups rows like `GROUP BY`
    /// (datatype3.html §7.1), so a `PARTITION BY x` where `x` is a NOCASE column, or an
    /// explicit `PARTITION BY x COLLATE NOCASE`, folds text when forming partitions rather
    /// than splitting bytewise. Resolved once at compile time; applied by the window
    /// operator's partition-keying (`ops::window`).
    pub partition_collations: Vec<Collation>,
    /// `ORDER BY` within each partition.
    pub order_by: Vec<SortKey>,
    /// The frame each row's function evaluates over.
    pub frame: WindowFrame,
}

#[cfg(test)]
mod tests {
    use super::*;
    // `super::*` already brings the parent's `SeqScan`, `EvalExpr`, and `Affinity`
    // into scope; only `Value` is new to the test.
    use minisqlite_types::Value;

    // Compile-smoke: the plan tree is constructible from the shared vocabulary and
    // its `Debug` derives work (proving `Debug, Clone` are wired end to end). This is
    // a scaffold check, not a behavioral one — there is no planning/execution logic
    // to exercise yet.
    #[test]
    fn filter_over_seqscan_is_constructible_and_debuggable() {
        let scan = PlanNode::SeqScan(SeqScan {
            table: "t".to_string(),
            db: DbIndex::MAIN,
            column_count: 2,
        });
        let root = PlanNode::Filter { input: Box::new(scan), predicate: EvalExpr::Column(0) };
        let plan = Plan {
            root,
            result_columns: vec!["a".to_string(), "b".to_string()],
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let cloned = plan.clone();
        assert!(!format!("{plan:?}").is_empty());
        assert!(!format!("{cloned:?}").is_empty());
    }

    #[test]
    fn insert_over_values_is_constructible_and_debuggable() {
        let values =
            PlanNode::Values { rows: vec![vec![EvalExpr::Literal(Value::Integer(1))]] };
        let root = PlanNode::Insert(Insert {
            table: "t".to_string(),
            db: DbIndex::MAIN,
            column_count: 1,
            columns: None,
            source: Box::new(values),
            source_width: 1,
            column_affinities: vec![Affinity::Integer],
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: Vec::new(),
            rowid_source: None,
            upsert: None,
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        });
        let plan = Plan {
            root,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: true,
            generated: Vec::new(),
        };
        assert!(plan.mutates);
        assert!(!format!("{plan:?}").is_empty());
    }

    #[test]
    fn trigger_program_composes_and_rides_on_a_dml_node() {
        // The compiled-trigger IR builds from the shared vocabulary (a WHEN `EvalExpr`
        // and action `Plan`s) and is carried on a DML node's `triggers`. Scaffold check:
        // construct + Clone + Debug, and confirm the field survives a clone — there is
        // no firing logic to exercise yet.
        let action = Plan {
            root: PlanNode::Values { rows: vec![vec![EvalExpr::Literal(Value::Integer(1))]] },
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: true,
            generated: Vec::new(),
        };
        let prog = TriggerProgram {
            name: "trg".to_string(),
            timing: minisqlite_sql::TriggerTiming::After,
            when: Some(EvalExpr::Column(0)),
            actions: vec![action],
        };
        let insert = Insert {
            table: "t".to_string(),
            db: DbIndex::MAIN,
            column_count: 1,
            columns: None,
            source: Box::new(PlanNode::SingleRow),
            source_width: 0,
            column_affinities: vec![Affinity::Integer],
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: vec![prog],
            checks: Vec::new(),
            rowid_source: None,
            upsert: None,
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        };
        let cloned = insert.clone();
        assert_eq!(cloned.triggers.len(), 1);
        assert_eq!(cloned.triggers[0].actions.len(), 1, "the action Plan survives the clone");
        assert!(cloned.triggers[0].when.is_some());
        assert!(matches!(cloned.triggers[0].timing, minisqlite_sql::TriggerTiming::After));
        assert!(!format!("{insert:?}").is_empty());
    }

    #[test]
    fn check_constraint_composes_and_rides_on_a_dml_node() {
        // A compiled CHECK is a bound predicate + a detail string; it clones and Debugs,
        // and survives on a DML node's `checks`. Scaffold check only — the executor
        // (separate owner) evaluates it; there is no evaluation logic here.
        let check = CheckConstraint {
            expr: EvalExpr::Column(0),
            detail: "t".to_string(),
        };
        let insert = Insert {
            table: "t".to_string(),
            db: DbIndex::MAIN,
            column_count: 1,
            columns: None,
            source: Box::new(PlanNode::SingleRow),
            source_width: 0,
            column_affinities: vec![Affinity::Integer],
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: vec![check],
            rowid_source: None,
            upsert: None,
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        };
        let cloned = insert.clone();
        assert_eq!(cloned.checks.len(), 1, "the check survives the clone");
        assert_eq!(cloned.checks[0].detail, "t");
        assert!(!format!("{insert:?}").is_empty());
    }
}
