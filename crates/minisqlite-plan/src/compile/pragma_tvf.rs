//! Compiling the `pragma_*` schema-introspection table-valued functions
//! (`pragma_table_info`, `pragma_index_list`, …) that appear in a FROM clause
//! (pragma.html §2 "PRAGMA functions").
//!
//! This is the pragma-TVF half of the FROM-clause TVF wiring; the JSON TVFs
//! (`json_each`/`json_tree`) live in [`crate::compile::from`]. The two are told apart by
//! function name and share the same two-phase shape:
//!
//! * **Phase 1** ([`from::resolve_table`](crate::compile::from)): a pragma TVF resolves to a
//!   [`Source::Derived`] carrying that PRAGMA's FIXED columns ([`pragma_columns`]) — the
//!   same shape a derived table uses (no rowid, own [`SynthCol`] list), so the rest of the
//!   binder treats it like any other derived source.
//! * **Phase 2** ([`from::build_source_leaf`](crate::compile::from)): the access leaf is a
//!   [`PlanNode::PragmaFunctionScan`] built by [`compile`], whose object-name / schema
//!   argument expressions are bound HERE against the whole FROM scope (pragma TVFs are
//!   implicitly LATERAL — pragma.html joins `pragma_index_list('t')` against
//!   `pragma_index_info(il.name)` — so an argument may reference a preceding table's
//!   columns, which the executor threads as the leaf's `outer` row).
//!
//! The rows themselves are produced at EXEC time by `minisqlite-catalog`'s shared
//! `pragma_rows`, the SAME builder the `PRAGMA` statement form uses, so a TVF and its
//! statement cannot diverge. The column NAMES / count also come from `minisqlite-catalog`
//! ([`PragmaFunction::column_names`]), so Phase 1's schema and the executor's row width
//! agree by construction.

use minisqlite_catalog::PragmaFunction;
use minisqlite_sql::{Expr, QualifiedName};
use minisqlite_types::{affinity_of_declared_type, Collation, Error, Result};

use crate::bind::scope::SynthCol;
use crate::bind::{bind_expr, Scope, Source};
use crate::plan::PlanNode;
use crate::plan_ctx::PlanCtx;

/// Classify a FROM table-valued-function name as one of the `pragma_*` introspection TVFs,
/// or `None` for any other name (a JSON TVF or a genuine unknown, handled by the caller). A
/// schema-qualified name is never one of these built-ins (a pragma TVF passes its schema as
/// the LAST ARGUMENT, `pragma_table_info('t','main')`, never as a `db.fn` qualifier), so a
/// qualifier disqualifies it — mirroring the JSON TVF classifier.
pub(crate) fn classify(name: &QualifiedName) -> Option<PragmaFunction> {
    if name.schema.is_some() {
        return None;
    }
    PragmaFunction::from_tvf_name(&name.name)
}

/// The Phase-1 derived-source schema for a pragma TVF: one visible column per name in the
/// PRAGMA's fixed column list ([`PragmaFunction::column_names`]), each with NONE affinity
/// (`affinity_of_declared_type(None)` = `Blob`) and BINARY collation. The pragma columns are
/// executor-produced values with no declared type, so — like a `VALUES` row or the JSON
/// TVF's columns — they carry no affinity of their own. There are NO hidden columns (unlike
/// the JSON TVFs' `json`/`root`): the object-name and schema are ordinary function arguments,
/// not hidden input columns.
pub(crate) fn pragma_columns(kind: PragmaFunction) -> Vec<SynthCol> {
    kind.column_names().iter().map(|name| pragma_synth_col(name)).collect()
}

/// One NONE-affinity / BINARY-collation visible synthetic column. (A local copy of
/// `from::synth_col`, which is private to that module — kept here so the pragma-TVF schema
/// owns its own construction rather than widening `from`'s surface.)
fn pragma_synth_col(name: &str) -> SynthCol {
    SynthCol {
        name: name.to_string(),
        affinity: affinity_of_declared_type(None),
        collation: Collation::Binary,
        hidden: false,
    }
}

/// Compile a pragma TVF FROM leaf into a [`PlanNode::PragmaFunctionScan`]. `kind` is the
/// already-classified pragma (from [`classify`]); `name` is carried only for error messages.
///
/// The object-name argument (`pragma_table_info('t')` → `'t'`) and the OPTIONAL trailing
/// schema argument (`pragma_table_info('t','main')` → `'main'`) are bound against the whole
/// FROM `scope` (implicitly LATERAL, as in Phase 2 for the JSON TVFs). Accepts 0, 1, or 2
/// arguments — the zero-argument form binds no name, yielding zero rows like the statement
/// form for an absent object; three or more is a loud error, matching SQLite. The carried
/// `column_count` is the Phase-1 schema width (the PRAGMA's fixed columns, no hidden slots),
/// so the emitted rows match the register width the outer scope already bound against.
pub(crate) fn compile(
    ctx: &mut PlanCtx,
    scope: &Scope,
    name: &QualifiedName,
    kind: PragmaFunction,
    args: &[Expr],
    src: &Source,
) -> Result<PlanNode> {
    if args.len() > 2 {
        return Err(Error::sql(format!(
            "wrong number of arguments to function {}()",
            name.name
        )));
    }
    let name_arg = match args.first() {
        Some(a) => Some(bind_expr(scope, ctx, a)?),
        None => None,
    };
    let schema_arg = match args.get(1) {
        Some(a) => Some(bind_expr(scope, ctx, a)?),
        None => None,
    };
    let column_count = src.width();
    debug_assert_eq!(
        column_count,
        kind.column_names().len(),
        "the pragma-TVF Phase-1 schema must be exactly the pragma's fixed columns"
    );
    Ok(PlanNode::PragmaFunctionScan { kind, name_arg, schema_arg, column_count })
}
