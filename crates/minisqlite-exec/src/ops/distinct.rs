//! `Distinct` — emit each distinct whole row once. Storage-class equality with
//! numeric value-folding (so `2` and `2.0` dedup together) and per-column collation
//! folding (so a `NOCASE` column folds `'a'`/`'A'`); see [`crate::keys`].
//!
//! Duplicate rows are compared with SQLite's `IS DISTINCT FROM` (lang_select.html §2.6),
//! i.e. each output column under its §7.1 collation. The planner resolves that collation
//! per output column and carries it on the node (`Distinct.column_collations`); this
//! operator only applies it. NULLs dedup together and numeric/BLOB dedup is unaffected
//! (collation only folds TEXT) — both are [`row_key`]'s contract.

use std::collections::HashSet;

use minisqlite_types::{Collation, Result, Row};

use crate::keys::{row_key, CellKey};
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a `DISTINCT` over `input`, deduping each output column under
/// `column_collations[i]` (one per column; a shorter list defaults the rest to BINARY,
/// per [`row_key`]'s contract).
pub(crate) fn distinct<'a>(
    input: Box<dyn RowCursor + 'a>,
    column_collations: &'a [Collation],
) -> Box<dyn RowCursor + 'a> {
    Box::new(DistinctCursor { input, seen: HashSet::new(), column_collations })
}

struct DistinctCursor<'a> {
    input: Box<dyn RowCursor + 'a>,
    /// The set of row keys already emitted. Grows with the number of DISTINCT rows —
    /// bounded by the result cardinality, the one materialization DISTINCT requires.
    seen: HashSet<Vec<CellKey>>,
    /// Per-output-column collating sequence for the duplicate comparison, resolved by
    /// the planner (borrowed from the plan node, so no per-row allocation).
    column_collations: &'a [Collation],
}

impl RowCursor for DistinctCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        // Pull until a not-yet-seen row appears or the input is exhausted.
        while let Some(row) = self.input.next_row(rt)? {
            if self.seen.insert(row_key(&row, self.column_collations)) {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
}
