//! Access-path leaves: the ways a plan reaches a base table's rows. These are the
//! bottom of every read tree, and where the planner's index-vs-scan choice is
//! recorded as data for the executor to run without re-deciding.
//!
//! Every leaf here scans one base table and emits that table's row in the shape the
//! shared ROW/REGISTER convention pins (see the module doc of [`crate::plan`]): a
//! table with `N` columns produces width `N+1`, `[c0, c1, …, c_{N-1}, rowid]`, with
//! the rowid as `Value::Integer` in register `N`. The `column_count` field on each
//! leaf carries `N` explicitly so the executor sizes the row as data rather than
//! re-deriving it from the catalog.

use minisqlite_expr::EvalExpr;
use minisqlite_types::DbIndex;

/// The order a scan walks its b-tree, and thus the order rows are emitted.
///
/// `Reverse` lets the planner satisfy a descending `ORDER BY` (on the rowid or on
/// an index's key order) by walking the tree backwards instead of adding a `Sort`.
#[derive(Debug, Clone)]
pub enum ScanDirection {
    /// Ascending key order (rowid ascending, or index key ascending).
    Forward,
    /// Descending key order (rowid descending, or index key descending).
    Reverse,
}

/// Full table scan of the table b-tree in ascending rowid order — the access path
/// of last resort, used when no rowid or index path covers the predicate. Cost is
/// proportional to the table size.
///
/// Emits `[c0, …, c_{N-1}, rowid]` for every row, width `column_count + 1`.
#[derive(Debug, Clone)]
pub struct SeqScan {
    /// The base table's name (plan nodes reference tables by name, never by page id;
    /// the executor resolves the root page through the catalog).
    pub table: String,
    /// Which database namespace the table lives in (`main`/`temp`/attached), so the
    /// executor opens the cursor on the right store. The binder resolves this from a
    /// schema qualifier or SQLite name-resolution search order; `DbIndex::MAIN` for the
    /// common single-store case.
    pub db: DbIndex,
    /// `N`, the table's declared column count (`CREATE TABLE` order). The emitted
    /// row width is `N + 1` (the trailing rowid register).
    pub column_count: usize,
}

/// One end of a rowid or index range: the bound value and whether it is inclusive
/// (`<=`/`>=`) or exclusive (`<`/`>`).
#[derive(Debug, Clone)]
pub struct RangeBound {
    /// The bound expression, evaluated once before the scan begins.
    pub value: EvalExpr,
    /// `true` for a closed bound (`>=` / `<=`), `false` for an open one (`>` / `<`).
    pub inclusive: bool,
}

/// How a [`RowidScan`] constrains the rowid it seeks on the table b-tree.
#[derive(Debug, Clone)]
pub enum RowidOp {
    /// Point lookup: `rowid = value`. Reaches at most one row via a single b-tree
    /// seek (`WHERE rowid = ?` or `WHERE <intpk-column> = ?`).
    Eq(EvalExpr),
    /// Range on the rowid: `lo <= rowid <= hi` (bounds inclusive/exclusive per each
    /// [`RangeBound`]). `None` on a side means that side is unbounded, so
    /// `{ lo: None, hi: None }` is a full ordered scan of the table b-tree.
    Range { lo: Option<RangeBound>, hi: Option<RangeBound> },
}

/// Rowid access path: seek or range-scan the table b-tree by its integer rowid key
/// (the `INTEGER PRIMARY KEY` / rowid path). Cost is proportional to the rows the
/// `op` selects, not the table size.
///
/// Like every table leaf it emits `[c0, …, c_{N-1}, rowid]`, width `column_count + 1`.
#[derive(Debug, Clone)]
pub struct RowidScan {
    pub table: String,
    /// Which database namespace the table lives in (see [`SeqScan::db`]).
    pub db: DbIndex,
    /// `N`, the table's column count; emitted width is `N + 1`.
    pub column_count: usize,
    /// The rowid constraint this path applies.
    pub op: RowidOp,
    /// Walk direction, so a descending order can be served without a `Sort`.
    pub direction: ScanDirection,
}

/// How an [`IndexScan`] positions on the index b-tree.
#[derive(Debug, Clone)]
pub enum IndexOp {
    /// Equality on a prefix of the index columns, optionally with a trailing range
    /// on the next column. `eq_prefix[k]` pins index column `k` to that value; `low`
    /// / `high` (when present) bound the column immediately after the equality
    /// prefix. An empty `eq_prefix` with a `low`/`high` is a pure leading-column
    /// range; an empty `eq_prefix` with no bounds is equivalent to a `FullScan`.
    Seek { eq_prefix: Vec<EvalExpr>, low: Option<RangeBound>, high: Option<RangeBound> },
    /// Full scan of the index in key order (e.g. to satisfy an `ORDER BY` on the
    /// index columns without a separate `Sort`, or to drive a covering read).
    FullScan,
}

/// Index access path: seek or scan the index b-tree, then — unless `covering` — fetch
/// each row from the table b-tree by the rowid stored in the index entry. Cost is
/// proportional to the entries the `op` selects.
///
/// Emits the table row `[c0, …, c_{N-1}, rowid]`, width `column_count + 1`, whether
/// or not the fetch is skipped: a covering index already carries every needed column
/// plus the rowid, so `covering` only elides the extra table lookup, never changes
/// the emitted shape.
#[derive(Debug, Clone)]
pub struct IndexScan {
    /// The base table the index is defined on (the row is fetched from here).
    pub table: String,
    /// Which database namespace the table and its index live in (see [`SeqScan::db`]).
    /// The index is always in the same namespace as its table (SQLite forbids an index
    /// across databases), so this names the store for BOTH b-tree cursors.
    pub db: DbIndex,
    /// `N`, the table's column count — the fetched row width before the `+1` rowid.
    pub column_count: usize,
    /// The index's name (its root page is resolved through the catalog).
    pub index: String,
    /// How this path positions on the index b-tree.
    pub op: IndexOp,
    /// Walk direction over the index, so index-order `ORDER BY` (ascending or
    /// descending) is served without a `Sort`.
    pub direction: ScanDirection,
    /// `true` when the index carries every column the query needs (a covering
    /// index): the executor reads values straight from the index entry and skips the
    /// table fetch. `false` requires the by-rowid table lookup per entry.
    pub covering: bool,
}
