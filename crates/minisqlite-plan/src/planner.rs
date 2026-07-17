//! The planning seam: [`Planner`]. It turns one parsed statement into an executable
//! [`Plan`], and it is where the access-path choice lives. Choosing an index seek
//! over a full table scan when an index covers a lookup or range is the planner's
//! job; the executor only runs the plan it is handed. The cost model and the richer
//! plan nodes live under this seam (in [`crate::plan`] / [`crate::access`]) and are
//! yours to grow — this file holds only the seam trait so the route cannot fork.

use minisqlite_catalog::Catalog;
use minisqlite_sql::Statement;
use minisqlite_types::Result;

use crate::plan::Plan;

/// The single planning seam. Declared exactly once, in this crate (enforced by
/// `minisqlite/tests/seams.rs`), so the route to planning cannot fork.
pub trait Planner {
    /// Compile ONE parsed statement into an executable plan, borrowing the catalog
    /// read-only to choose access paths. Planning never mutates schema or copies
    /// data; any schema/data change is expressed as a plan the executor runs.
    fn plan(&self, statement: &Statement, catalog: &dyn Catalog) -> Result<Plan>;
}
