//! GENERATED-column evaluation shared by the read (scan) and write (INSERT/UPDATE)
//! paths (`gencol.html`): compute a table's generated column values into a logical row,
//! and build the physical record that omits VIRTUAL columns.
//!
//! The generation programs themselves ([`GeneratedProgram`]) are bound once by the
//! planner and carried on [`Plan::generated`](minisqlite_plan::Plan); an operator looks
//! its table's programs up with
//! [`Plan::generated_programs`](minisqlite_plan::Plan::generated_programs) and passes the
//! slice here. This module is pure evaluation + record layout — it never touches the
//! plan map, so the read and write paths cannot drift on how a generated value is
//! computed or stored.

use minisqlite_catalog::TableDef;
use minisqlite_expr::eval;
use minisqlite_plan::GeneratedProgram;
use minisqlite_types::{apply_affinity, Result, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::row::is_virtual_generated;
use crate::runtime::Runtime;

/// Compute generated column values into `row`, in program order, applying each column's
/// declared affinity to the result (`gencol.html`: the value is coerced to the column's
/// datatype like an ordinary column).
///
/// `row` MUST be the logical row the generation expressions were bound against: for a
/// rowid table `[c0..c_{N-1}, rowid]` (width `N+1`, so an INTEGER PRIMARY KEY reference
/// resolves to register `N`); for a WITHOUT ROWID table `[c0..c_{N-1}]` (width `N`). Each
/// program writes `row[col_index]` (`col_index < N`), and the programs are in DEPENDENCY
/// order (topologically sorted by `compile::generated`, NOT column order), so a generated
/// column a later program references is already filled before that later program evaluates —
/// this is what lets one generated column reference another declared after it.
///
/// `only_virtual` selects the READ path (`true`: STORED columns are already present in the
/// decoded record, so recomputing them is skipped) versus the WRITE path (`false`: every
/// generated column, STORED and VIRTUAL, is (re)computed before the record is built and
/// constraints run). A generated expression is subquery/parameter-free (`gencol.html`
/// §2.3, enforced at bind time), so the `EvalCtx` here needs no real `outer` frame.
pub(crate) fn compute_generated(
    programs: &[GeneratedProgram],
    only_virtual: bool,
    row: &mut [Value],
    env: Env,
    rt: &mut Runtime,
) -> Result<()> {
    for prog in programs {
        if only_virtual && prog.stored {
            continue;
        }
        // Read the row (immutable) to evaluate, then write the result back — the two
        // borrows do not overlap. The `EvalCtx` reborrows `rt` each iteration (it is moved
        // into the context) and copies `env`.
        let value = {
            let mut ctx = EvalCtx { rt: &mut *rt, env, outer: &[] };
            eval(&prog.expr, row, &mut ctx)?
        };
        row[prog.col_index] = apply_affinity(value, prog.affinity);
    }
    Ok(())
}

/// Build the PHYSICAL record for a row of a table with generated columns: the non-virtual
/// columns in `CREATE TABLE` order, OMITTING every VIRTUAL generated column (which is
/// never stored — `gencol.html`), with the INTEGER PRIMARY KEY alias column stored as
/// `NULL` (its value is the b-tree key, not a record field — the same convention the plain
/// INSERT/UPDATE record building uses).
///
/// `logical` is the width-`N` (or wider — only `[0, N)` is read) logical row with every
/// generated value already computed by [`compute_generated`]; the returned vec is the
/// exact positional record a real `sqlite3` reads back (a VIRTUAL column leaves NO gap).
/// Used only when the table HAS a generated column — the ordinary record path
/// (`encode_record` over the affinity-applied `stored` vec) is untouched, so a table with
/// no generated column pays nothing.
pub(crate) fn stored_record(logical: &[Value], def: &TableDef) -> Vec<Value> {
    let mut out = Vec::with_capacity(def.columns.len());
    for (i, col) in def.columns.iter().enumerate() {
        if is_virtual_generated(col) {
            continue;
        }
        if def.rowid_alias == Some(i) {
            // The INTEGER PRIMARY KEY alias is stored NULL (the rowid lives in the b-tree
            // key and is refilled on read), matching the plain record path.
            out.push(Value::Null);
        } else {
            out.push(logical[i].clone());
        }
    }
    out
}
