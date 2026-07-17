//! The column schema a `CREATE TABLE ... AS SELECT` (CTAS) builds, derived from the
//! SELECT alone (`lang_createtable.html` §2.1). The engine intercepts CTAS and
//! synthesizes a plain `Columns` table plus an `INSERT ... SELECT`; this module owns the
//! one non-trivial derivation: turning the SELECT's result set into `(name, decl_type)`
//! pairs.
//!
//! Two independent facts per column, from different §2.1 rules:
//! - NAME + COUNT (authoritative): "the same number of columns as the SELECT returns …
//!   the name of each column is the same as the name of the corresponding column in the
//!   result set". These come from [`compile_select`]'s result names — the SAME names and
//!   width the `INSERT ... SELECT` will populate, so the created columns cannot drift
//!   from the rows that fill them.
//! - DECLARED TYPE (best-effort): "determined by the expression affinity of the
//!   corresponding expression" via the §2.1 affinity→type table, with affinity per
//!   datatype3 §3.2. Only a simple `SELECT <cols> FROM <src>` has typed columns (a bare
//!   column ref → that column's affinity, `CAST(e AS T)` → T's affinity, everything else
//!   → none); a compound / VALUES / WITH body degrades to all-"" declared types, which
//!   is exactly the §2.1 BLOB/NONE row.

use minisqlite_catalog::Catalog;
use minisqlite_functions::FunctionRegistry;
use minisqlite_sql::{FromClause, ResultColumn, Select, SelectBody, SelectCore};
use minisqlite_types::{Affinity, Result};

use crate::bind::expr::operand_affinity;
use crate::bind::Scope;
use crate::compile::from::resolve_from;
use crate::compile::select::compile_select;
use crate::plan_ctx::PlanCtx;

/// One column of the table a CTAS builds: the output name the SELECT's result set gives
/// it, and the declared type the §2.1 affinity→type mapping assigns. `decl_type` is
/// `None` for the empty declared type "" (the BLOB / no-affinity row of the table),
/// which `PRAGMA table_info` renders as the empty string.
pub struct CtasColumn {
    pub name: String,
    pub decl_type: Option<String>,
}

/// Derive the CTAS table's column schema from `sel` (its names/count and each column's
/// declared type) per `lang_createtable.html` §2.1. `catalog` resolves the SELECT's
/// source tables for both the name compile and the affinity probe.
///
/// A genuinely invalid SELECT (an unresolvable column, a bad function) surfaces here as
/// the CTAS error — the engine runs this BEFORE creating the table, so a bad SELECT
/// leaves no table (the §2.1 "CREATE + populate is atomic" guarantee starts here).
pub fn ctas_columns(sel: &Select, catalog: &dyn Catalog) -> Result<Vec<CtasColumn>> {
    // NAMES + COUNT (authoritative): compile the SELECT for its result names. The plan
    // node is discarded — only the names matter — but compiling it is also what makes an
    // invalid SELECT fail here rather than after a half-built table.
    let registry = FunctionRegistry::builtins();
    let mut ctx = PlanCtx::new(&registry, catalog);
    let (_node, names) = compile_select(&mut ctx, sel)?;

    // DECLARED TYPES (best-effort): the §2.1 mapping of each result expression's
    // affinity, in the same order/width as `names`. Non-simple shapes and any resolution
    // difficulty degrade to all-"" — never an error, since `names` already fixed the
    // real contract.
    let affinities = ctas_affinities(sel, catalog, names.len());
    debug_assert_eq!(affinities.len(), names.len(), "one affinity slot per result name");

    Ok(names
        .into_iter()
        .zip(affinities)
        .map(|(name, affinity)| CtasColumn { name, decl_type: decl_type_of(affinity) })
        .collect())
}

/// Best-effort per-column affinity for the simple `SELECT <cols> FROM <src>` shape, in
/// result order and length `want`. A body we do not type (a leading `WITH`, or a
/// compound / VALUES core), a column set whose affinity width disagrees with `want`
/// (e.g. a `*` the affinity probe and the name compile expanded differently), or any
/// name-resolution error all yield `vec![None; want]` — every column the empty declared
/// type "", the §2.1 BLOB/NONE default.
fn ctas_affinities(sel: &Select, catalog: &dyn Catalog, want: usize) -> Vec<Option<Affinity>> {
    let typed = if sel.with.is_none() {
        match &sel.body {
            SelectBody::Select(SelectCore::Query { columns, from, .. }) => {
                query_affinities(catalog, columns, from).ok()
            }
            _ => None,
        }
    } else {
        None
    };
    match typed {
        Some(affinities) if affinities.len() == want => affinities,
        _ => vec![None; want],
    }
}

/// Resolve each result column's affinity against the FROM sources, in result order. A
/// bare `expr` uses [`operand_affinity`] (a column ref → its affinity, `CAST(e AS T)` →
/// T's affinity, an operator / function / literal → none); a `*` / `table.*` expands to
/// its source columns, each a bare column reference carrying that column's own affinity.
fn query_affinities(
    catalog: &dyn Catalog,
    columns: &[ResultColumn],
    from: &Option<FromClause>,
) -> Result<Vec<Option<Affinity>>> {
    let (sources, coalesced) = resolve_from(catalog, from, 0)?;
    let scope = Scope {
        sources: &sources,
        coalesced: &coalesced,
        parent: None,
        grouping: None,
        saw_correlated: None,
        correlated_cols: None,
        nondeterministic: None,
        windowing: None,
    };
    let mut out = Vec::with_capacity(columns.len());
    for col in columns {
        match col {
            ResultColumn::Expr { expr, .. } => out.push(operand_affinity(&scope, expr)?),
            ResultColumn::Star => push_star_affinities(&scope, None, &mut out)?,
            ResultColumn::TableStar(table) => push_star_affinities(&scope, Some(table), &mut out)?,
        }
    }
    Ok(out)
}

/// Append the affinity of each column a `*` (`table = None`) or `table.*` expands to.
/// A star column is a bare column reference, so it carries that column's own affinity
/// (an `INTEGER PRIMARY KEY` alias reads as integer, exactly as a bare reference to it
/// would); expansion order matches [`Scope::expand_star`], which the name compile used.
fn push_star_affinities(
    scope: &Scope,
    table: Option<&str>,
    out: &mut Vec<Option<Affinity>>,
) -> Result<()> {
    for (_reg, name) in scope.expand_star(table)? {
        out.push(Some(scope.resolve_column(None, &name)?.affinity));
    }
    Ok(())
}

/// The §2.1 declared-type string for an expression affinity: TEXT→"TEXT",
/// NUMERIC→"NUM", INTEGER→"INT", REAL→"REAL", and BLOB / no-affinity → `None` (the
/// empty declared type ""). This is the exact mapping table in `lang_createtable.html`
/// §2.1; BLOB affinity and "no affinity" share the empty-string row.
fn decl_type_of(affinity: Option<Affinity>) -> Option<String> {
    match affinity {
        Some(Affinity::Text) => Some("TEXT".to_string()),
        Some(Affinity::Numeric) => Some("NUM".to_string()),
        Some(Affinity::Integer) => Some("INT".to_string()),
        Some(Affinity::Real) => Some("REAL".to_string()),
        Some(Affinity::Blob) | None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §2.1 affinity→declared-type table, exhaustively over all five affinities plus
    /// the no-affinity case. BLOB and "no affinity" both map to the empty declared type
    /// (`None`); the other four map to their fixed spellings. An exhaustive match, so a
    /// new `Affinity` variant would fail to compile until mapped here.
    #[test]
    fn decl_type_mapping_covers_every_affinity() {
        assert_eq!(decl_type_of(Some(Affinity::Text)).as_deref(), Some("TEXT"));
        assert_eq!(decl_type_of(Some(Affinity::Numeric)).as_deref(), Some("NUM"));
        assert_eq!(decl_type_of(Some(Affinity::Integer)).as_deref(), Some("INT"));
        assert_eq!(decl_type_of(Some(Affinity::Real)).as_deref(), Some("REAL"));
        assert_eq!(decl_type_of(Some(Affinity::Blob)), None);
        assert_eq!(decl_type_of(None), None);
    }
}
