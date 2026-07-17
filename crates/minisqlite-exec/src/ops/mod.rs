//! The read operators, one streaming [`RowCursor`](crate::RowCursor) per file so a
//! later feature lands in its own cell rather than contending here.
//!
//! Each operator holds its [`Env`](crate::env::Env) plus its own small state, pulls
//! from its input(s) one row at a time, and never materializes an intermediate
//! unless it provably must (only [`sort`] and [`distinct`] buffer, and both are
//! bounded by the input size). On a malformed plan an operator returns an `Err`; it
//! never panics, and every loop it runs is bounded by its input.
//!
//! ## The `outer` plumbing (for correlated subqueries)
//! A leaf operator emits `outer ++ local_row`; when `outer` is empty (the top level) that
//! is just `local_row` at no cost. This holds for EVERY FROM leaf: [`scan`] / rowid-scan /
//! [`indexscan`] (via [`with_outer`]), [`values`] / single-row, [`table_function`],
//! [`pragma_function`], and
//! [`cte`] — a derived table / view / CTE reference, which drains its STANDALONE base-0
//! body with an empty outer and prepends the prefix once at its output boundary.
//! ([`recursive_scan`](cte::recursive_scan) is NOT a correlated leaf: it reads a recursive
//! step's working-table frame, which stays at `column_count` width inside a step that is
//! itself drained standalone.) An interior operator (filter / project / sort / limit /
//! distinct / [`join`] / [`aggregate`]) forwards the SAME `outer` to its child's build and
//! does NOT prepend — the leaf below it already did — and at each eval site it uses the
//! *current row* as the `outer` a nested subquery would see. In a [`join`] only the LEFT
//! spine carries `outer`; the right child is built with an empty outer so the prefix
//! appears exactly once. `outer` is empty at the top level and non-empty for a correlated
//! subplan (`context::open_subquery`) or a firing trigger action (whose `OLD ++ NEW` frame
//! is threaded as `outer` through `build_dml`).

pub(crate) mod aggregate;
pub(crate) mod cte;
pub(crate) mod distinct;
pub(crate) mod filter;
pub(crate) mod indexscan;
pub(crate) mod join;
pub(crate) mod limit;
pub(crate) mod minmax_seek;
pub(crate) mod pragma_function;
pub(crate) mod project;
pub(crate) mod scan;
pub(crate) mod setop;
pub(crate) mod sort;
pub(crate) mod table_function;
pub(crate) mod values;
pub(crate) mod window;

// Shared `ORDER BY` comparator (`sortkey::compare_sort_keys`) — not a streaming
// operator but the one set of NULL/collation/direction rules that `sort` and
// aggregate `ORDER BY` (and, later, the window operator) share so they cannot drift.
pub(crate) mod sortkey;

// DML write operators: apply a mutation through the exclusive `&mut Pager`, then
// stream any `RETURNING` rows — not streaming reads like the operators above.
pub(crate) mod delete;
pub(crate) mod insert;
pub(crate) mod update;

// RETURNING-clause evaluation shared by every DML write operator (one home so the
// cross-namespace read view a RETURNING subquery needs is defined once — INSERT, UPDATE,
// and DELETE cannot drift back to a single-namespace assumption).
pub(crate) mod returning;

// View DML redirected through `INSTEAD OF` triggers: a view owns no b-tree, so this makes
// NO base write — it drives each affected view row's `OLD ++ NEW` frame through the view's
// matching `INSTEAD OF` programs (reusing `trigger::fire_program`).
pub(crate) mod instead_of;

// Trigger firing shared by every DML write operator: build the OLD/NEW register frame
// for one row and run the pre-compiled trigger programs (their WHEN gate + action plans),
// so the frame layout and fire loop have one home and INSERT/UPDATE/DELETE cannot drift.
pub(crate) mod trigger;

// Index maintenance shared by every DML write operator (one home for index-key
// encoding + the UNIQUE probe, so the index and table never disagree).
pub(crate) mod dml_index;

// WITHOUT ROWID storage: the PRIMARY KEY index-b-tree layout, encode/decode, and PK
// uniqueness probe shared by the WR scan and INSERT paths (one home so read and write
// cannot disagree on how a WR row is stored or ordered).
pub(crate) mod without_rowid;

// Column-constraint enforcement shared by the DML write operators (one home for the
// NOT NULL semantics, so INSERT and UPDATE cannot drift).
pub(crate) mod constraints;

// GENERATED-column evaluation shared by the read (scan) and write (INSERT/UPDATE) paths:
// compute a table's generated values into a logical row, and build the physical record
// that omits VIRTUAL columns (`gencol.html`). One home so read and write cannot disagree
// on how a generated value is computed or which columns are stored.
pub(crate) mod generated;

// SQLite's `OP_MustBeInt` integer coercion ("apply NUMERIC affinity, then require an
// integer, else datatype mismatch"), shared by every operator that needs it — the
// LIMIT/OFFSET bounds and the INTEGER PRIMARY KEY (rowid-alias) column INSERT/UPDATE
// write (a recursive-CTE body LIMIT/OFFSET bound reaches it indirectly: the planner
// lowers it to a `PlanNode::Limit`, so it coerces through `limit`) — so the one rule and
// its error string cannot drift.
pub(crate) mod must_be_int;

// AUTOINCREMENT rowid bookkeeping over the internal `sqlite_sequence(name, seq)` table —
// the read+write of a table's "largest rowid ever used" high-water the INSERT write path
// consults so an AUTOINCREMENT rowid is monotonic and never reused after a delete.
pub(crate) mod sequence;

// FOREIGN KEY enforcement (child side) shared by INSERT and UPDATE: verify a new/updated
// child row references an existing parent when `PRAGMA foreign_keys` is ON (no-op when
// off or the table declares no FK). Parent-side ON DELETE/UPDATE actions and the deferred
// COMMIT recheck live here too.
pub(crate) mod foreign_key;

use minisqlite_types::{Row, Value};

/// Prepend the `outer` row to a leaf's `local` row (`outer ++ local`). Empty `outer`
/// (the common top-level case) returns `local` unchanged with no allocation or copy;
/// otherwise the outer columns (cloned) come first, then `local`.
pub(crate) fn with_outer(outer: &[Value], mut local: Row) -> Row {
    if outer.is_empty() {
        return local;
    }
    let mut row = Vec::with_capacity(outer.len() + local.len());
    row.extend(outer.iter().cloned());
    row.append(&mut local);
    row
}
