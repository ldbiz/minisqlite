//! `ALTER TABLE DROP COLUMN` eligibility checks over the parsed `CREATE TABLE`.
//!
//! SQLite removes a column by deleting its definition from the stored `CREATE TABLE`
//! text and re-parsing the whole schema; the drop succeeds only if nothing that
//! *survives* still references the column (`lang_altertable.html` §5, §5.1). This
//! module is the pure decision core of that rule: given the parsed table and the
//! column, it reports whether the drop is blocked and why, with no catalog or pager
//! state, so it is exhaustively unit-testable.
//!
//! Split of responsibility: the definition-level restrictions (the column being a
//! PRIMARY KEY / UNIQUE, or a surviving CHECK / generated expression / table
//! constraint naming it) live here; the *index* restrictions — the column being
//! indexed by an explicit index or named in a partial index's `WHERE` (§5 bullets
//! 3-4) — are checked by the caller in `schemacatalog`, which owns the index set.
//! References from triggers / views (§5 bullet 8) are an UNCHECKED gap, not a loud
//! deferral: nothing here (or in the caller) inspects them, so a DROP COLUMN that a
//! trigger or view still references silently succeeds and leaves that object dangling
//! (it breaks only when later re-parsed/expanded). Faithful enforcement needs a
//! scope-aware binder to resolve which table a reference belongs to — a routed
//! follow-up, beyond this crate's enumerated restriction set.

use minisqlite_sql::{
    ColumnConstraintKind, CreateTable, CreateTableBody, Expr, FrameBound, FunctionArgs,
    IndexedColumn, IndexedColumnTarget, InRhs, OverClause, TableConstraintKind, WindowFrame,
};

/// Whether `expr` references the column named `col` (ASCII case-insensitive) anywhere
/// in its tree. Used to test surviving CHECK constraints, generated-column
/// expressions, and partial-index `WHERE` clauses for a mention of the dropped column.
///
/// The walk descends every sub-expression. A nested `SELECT` (a scalar subquery,
/// `EXISTS`, or `IN (SELECT …)`) is the one construct whose inner AST we do NOT walk —
/// but it is treated as a reference (returns `true`), the fail-CLOSED choice the DROP
/// COLUMN contract requires ("Fail CLOSED (reject) if a reference cannot be analyzed").
/// This engine's parser DOES accept a subquery inside a CHECK / generated /
/// partial-index expression — CHECK and generated columns parse via the general
/// `parse_expr` with no post-parse rejection — so such a constraint can be stored
/// verbatim on page 1. Rather than analyze the nested SELECT's column list, we
/// conservatively assume it MIGHT name `col` and block the drop; real sqlite likewise
/// refuses the drop here. Over-blocking a subquery that does not actually reference
/// `col` is the safe direction: it declines a rare drop, never permits one that leaves
/// a surviving constraint pointing at a now-missing column (a non-reparseable schema).
pub(crate) fn expr_references_column(expr: &Expr, col: &str) -> bool {
    match expr {
        Expr::Column { name, .. } => name.eq_ignore_ascii_case(col),
        Expr::Literal(_) | Expr::BindParam(_) | Expr::Raise(_) => false,
        Expr::Unary { expr, .. } => expr_references_column(expr, col),
        Expr::Binary { left, right, .. } => {
            expr_references_column(left, col) || expr_references_column(right, col)
        }
        Expr::Function { args, filter, over, order_by, .. } => {
            let in_args = match args {
                FunctionArgs::List(list) => list.iter().any(|e| expr_references_column(e, col)),
                FunctionArgs::Star | FunctionArgs::Empty => false,
            };
            in_args
                || filter.as_deref().is_some_and(|e| expr_references_column(e, col))
                // An aggregate's in-argument ORDER BY (`group_concat(x ORDER BY col)`) can
                // hide a column reference too; scan it for the same fail-closed reason.
                || order_by.iter().any(|t| expr_references_column(&t.expr, col))
                // `over` is walked for the same fail-closed reason as `filter`: a window
                // function cannot *validly* appear in a CHECK / generated / partial-index
                // expression, but if a lenient parse ever yielded one we must not miss a
                // column hidden only in its PARTITION BY / ORDER BY / frame bounds and
                // wrongly permit the drop. (A nested SELECT is the one construct we do NOT
                // walk into — it is instead treated as a reference, the same fail-closed
                // end; see the doc above.)
                || over.as_ref().is_some_and(|o| over_clause_references_column(o, col))
        }
        Expr::Cast { expr, .. } => expr_references_column(expr, col),
        Expr::Collate { expr, .. } => expr_references_column(expr, col),
        Expr::Like { lhs, rhs, escape, .. } => {
            expr_references_column(lhs, col)
                || expr_references_column(rhs, col)
                || escape.as_deref().is_some_and(|e| expr_references_column(e, col))
        }
        Expr::Between { expr, low, high, .. } => {
            expr_references_column(expr, col)
                || expr_references_column(low, col)
                || expr_references_column(high, col)
        }
        Expr::In { expr, rhs, .. } => {
            expr_references_column(expr, col)
                || match rhs {
                    InRhs::List(list) => list.iter().any(|e| expr_references_column(e, col)),
                    InRhs::Table { args, .. } => args.iter().any(|e| expr_references_column(e, col)),
                    // `IN (SELECT …)`: fail closed — a subquery we do not walk is treated
                    // as possibly referencing `col`, blocking the drop (see doc above).
                    InRhs::Select(_) => true,
                }
        }
        Expr::Case { operand, whens, else_expr } => {
            operand.as_deref().is_some_and(|e| expr_references_column(e, col))
                || whens
                    .iter()
                    .any(|(w, t)| expr_references_column(w, col) || expr_references_column(t, col))
                || else_expr.as_deref().is_some_and(|e| expr_references_column(e, col))
        }
        Expr::IsNull(e) | Expr::NotNull(e) => expr_references_column(e, col),
        Expr::Parenthesized(list) => list.iter().any(|e| expr_references_column(e, col)),
        // A nested SELECT is not walked; treat it as a reference (fail closed) so a
        // dropped column hidden inside a surviving subquery still blocks the drop rather
        // than corrupting the schema (see doc above; real sqlite refuses here too).
        Expr::Exists { .. } | Expr::Subquery(_) => true,
    }
}

/// Whether a window function's `OVER` clause names `col` in its `PARTITION BY`,
/// `ORDER BY`, or frame-bound offset expressions. See the fail-closed rationale at the
/// `Expr::Function` arm of [`expr_references_column`].
fn over_clause_references_column(over: &OverClause, col: &str) -> bool {
    let spec = match over {
        // `OVER window-name` carries no expressions of its own.
        OverClause::WindowName(_) => return false,
        OverClause::Spec(spec) => spec,
    };
    spec.partition_by.iter().any(|e| expr_references_column(e, col))
        || spec.order_by.iter().any(|t| expr_references_column(&t.expr, col))
        || spec.frame.as_ref().is_some_and(|f| frame_references_column(f, col))
}

/// Whether either bound of a window frame is a `col`-referencing offset expression
/// (`<expr> PRECEDING` / `<expr> FOLLOWING`); the unbounded and `CURRENT ROW` bounds
/// carry no expression.
fn frame_references_column(frame: &WindowFrame, col: &str) -> bool {
    let bound_refs = |b: &FrameBound| match b {
        FrameBound::Preceding(e) | FrameBound::Following(e) => expr_references_column(e, col),
        FrameBound::UnboundedPreceding
        | FrameBound::CurrentRow
        | FrameBound::UnboundedFollowing => false,
    };
    bound_refs(&frame.start) || frame.end.as_ref().is_some_and(bound_refs)
}

/// If dropping column `col` (0-based `drop_idx` in the column list) is forbidden by the
/// table's own `CREATE TABLE` definition, return `Some(reason)`; otherwise `None`.
///
/// Implements SQLite's definition-level rule: the dropped column's OWN constraints go
/// away with it and never block (spec: a CHECK "not associated with the column being
/// dropped"), with the two hard exceptions that the column may be neither a PRIMARY KEY
/// nor UNIQUE; and no *surviving* part of the `CREATE TABLE` — another column's CHECK or
/// generated expression, or any table-level constraint — may still name the column.
/// Index and partial-index restrictions are the caller's (it owns the index set).
pub(crate) fn create_table_drop_block(
    create: &CreateTable,
    col: &str,
    drop_idx: usize,
) -> Option<String> {
    let (columns, constraints) = match &create.body {
        CreateTableBody::Columns { columns, constraints, .. } => (columns, constraints),
        // A CREATE TABLE ... AS SELECT has no alterable column list here; the caller
        // decides what to do, so this check imposes nothing.
        CreateTableBody::AsSelect(_) => return None,
    };

    // The dropped column's own PRIMARY KEY / UNIQUE: removed with the column, yet SQLite
    // still forbids dropping such a column (§5 bullets 1-2).
    if let Some(dropped) = columns.get(drop_idx) {
        for c in &dropped.constraints {
            match c.kind {
                ColumnConstraintKind::PrimaryKey { .. } => {
                    return Some(format!("cannot drop column \"{col}\": it is a PRIMARY KEY"));
                }
                ColumnConstraintKind::Unique { .. } => {
                    return Some(format!(
                        "cannot drop column \"{col}\": it has a UNIQUE constraint"
                    ));
                }
                // Every other column constraint goes away *with* the column and never
                // blocks its own drop (a NOT NULL / NULL / DEFAULT / COLLATE / column FK,
                // or a CHECK / generated expr on the dropped column itself — §5.1). Named
                // exhaustively so a new `ColumnConstraintKind` forces a decision here.
                ColumnConstraintKind::NotNull { .. }
                | ColumnConstraintKind::Null { .. }
                | ColumnConstraintKind::Check(_)
                | ColumnConstraintKind::Default(_)
                | ColumnConstraintKind::Collate(_)
                | ColumnConstraintKind::ForeignKey(_)
                | ColumnConstraintKind::Generated { .. } => {}
            }
        }
    }

    // A CHECK or generated expression on ANOTHER surviving column that still names the
    // dropped one. The dropped column's own constraints (i == drop_idx) are skipped:
    // they are removed with the column.
    for (i, c) in columns.iter().enumerate() {
        if i == drop_idx {
            continue;
        }
        for cons in &c.constraints {
            let refs = match &cons.kind {
                // Only a surviving CHECK or generated expression can *name* another
                // column; if it names the dropped one, the drop is blocked.
                ColumnConstraintKind::Check(expr)
                | ColumnConstraintKind::Generated { expr, .. } => expr_references_column(expr, col),
                // The remaining kinds constrain only their own column and never name a
                // sibling (a SQLite DEFAULT expression may not reference other columns; a
                // column-level FK targets another *table*). Exhaustive so a future
                // column-referencing kind can't be silently ignored.
                ColumnConstraintKind::PrimaryKey { .. }
                | ColumnConstraintKind::Unique { .. }
                | ColumnConstraintKind::NotNull { .. }
                | ColumnConstraintKind::Null { .. }
                | ColumnConstraintKind::Default(_)
                | ColumnConstraintKind::Collate(_)
                | ColumnConstraintKind::ForeignKey(_) => false,
            };
            if refs {
                return Some(format!(
                    "cannot drop column \"{col}\": referenced by column \"{}\"",
                    c.name
                ));
            }
        }
    }

    // Any surviving table-level constraint that still names the dropped column. The
    // match is exhaustive (no `_`) so a future column-referencing `TableConstraintKind`
    // variant forces a decision here instead of being silently absorbed — the same
    // add-a-variant-breaks-the-build discipline as the column-level match above. Each
    // arm reports its own reason iff it references `col` (a guard can't compose with
    // exhaustiveness in Rust, so the test moves inside the arm).
    for tc in constraints {
        let reason = match &tc.kind {
            TableConstraintKind::PrimaryKey { columns: cols, .. } => {
                indexed_columns_reference(cols, col).then_some("it is part of the PRIMARY KEY")
            }
            TableConstraintKind::Unique { columns: cols, .. } => {
                indexed_columns_reference(cols, col).then_some("it has a UNIQUE constraint")
            }
            TableConstraintKind::Check(expr) => {
                expr_references_column(expr, col).then_some("it is used in a CHECK constraint")
            }
            TableConstraintKind::ForeignKey { columns: cols, .. } => cols
                .iter()
                .any(|c| c.eq_ignore_ascii_case(col))
                .then_some("it is used in a FOREIGN KEY constraint"),
        };
        if let Some(reason) = reason {
            return Some(format!("cannot drop column \"{col}\": {reason}"));
        }
    }

    None
}

/// Whether any entry in an indexed-column list (a table `PRIMARY KEY`/`UNIQUE`) names
/// `col` (case-insensitive) or is an expression that references it.
fn indexed_columns_reference(cols: &[IndexedColumn], col: &str) -> bool {
    cols.iter().any(|ic| match &ic.target {
        IndexedColumnTarget::Name(name) => name.eq_ignore_ascii_case(col),
        IndexedColumnTarget::Expr(expr) => expr_references_column(expr, col),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_sql::{parse, Statement};

    /// Parse one `CREATE TABLE` and return it, panicking on any other shape.
    fn create(sql: &str) -> CreateTable {
        match parse(sql).unwrap().statements.as_slice() {
            [Statement::CreateTable(ct)] => (**ct).clone(),
            other => panic!("expected one CREATE TABLE, got {other:?}"),
        }
    }

    /// The 0-based index of column `name` in a parsed `CREATE TABLE`.
    fn idx(ct: &CreateTable, name: &str) -> usize {
        match &ct.body {
            CreateTableBody::Columns { columns, .. } => columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(name))
                .unwrap_or_else(|| panic!("no column {name}")),
            other => panic!("expected a column body, got {other:?}"),
        }
    }

    /// `create_table_drop_block` for `col` in `sql`.
    fn block(sql: &str, col: &str) -> Option<String> {
        let ct = create(sql);
        create_table_drop_block(&ct, col, idx(&ct, col))
    }

    // --- expr_references_column ------------------------------------------------

    fn check_expr(sql: &str) -> Expr {
        // Parse a CHECK constraint's expression out of a one-column table.
        let ct = create(&format!("CREATE TABLE t(a, b CHECK({sql}))"));
        match &ct.body {
            CreateTableBody::Columns { columns, .. } => {
                for cons in &columns[1].constraints {
                    if let ColumnConstraintKind::Check(e) = &cons.kind {
                        return e.clone();
                    }
                }
                panic!("no CHECK expr parsed from {sql}");
            }
            other => panic!("expected a column body, got {other:?}"),
        }
    }

    #[test]
    fn expr_ref_finds_column_in_varied_positions() {
        assert!(expr_references_column(&check_expr("a > 0"), "a"));
        assert!(expr_references_column(&check_expr("A > 0"), "a")); // case-insensitive
        assert!(expr_references_column(&check_expr("length(a) > 1"), "a")); // function arg
        assert!(expr_references_column(&check_expr("a BETWEEN 1 AND 9"), "a"));
        assert!(expr_references_column(&check_expr("a IN (1, 2, 3)"), "a"));
        assert!(expr_references_column(&check_expr("CASE WHEN a THEN 1 ELSE 0 END"), "a"));
        assert!(expr_references_column(&check_expr("(a + 1) * 2 > 0"), "a"));
        assert!(expr_references_column(&check_expr("CAST(a AS INT) > 0"), "a"));
    }

    #[test]
    fn expr_ref_finds_column_through_unary_like_isnull_and_collate() {
        // The Unary / Like / IsNull / NotNull / Collate arms (a valid CHECK can use any
        // of these against the column) must each report the reference.
        assert!(expr_references_column(&check_expr("NOT a"), "a")); // Unary
        assert!(expr_references_column(&check_expr("a LIKE 'x%'"), "a")); // Like
        assert!(expr_references_column(&check_expr("a LIKE 'x%' ESCAPE a"), "a")); // Like escape
        assert!(expr_references_column(&check_expr("a ISNULL"), "a")); // IsNull
        assert!(expr_references_column(&check_expr("a NOTNULL"), "a")); // NotNull
        assert!(expr_references_column(&check_expr("a COLLATE nocase = 'x'"), "a")); // Collate
    }

    #[test]
    fn expr_ref_finds_column_inside_a_window_over_clause() {
        // A window function is not *valid* in a CHECK, but the walk is fail-closed: if a
        // lenient parse yields one, a column hidden only in the OVER clause must still be
        // seen so the drop is (conservatively) blocked.
        assert!(expr_references_column(&check_expr("count(*) OVER (ORDER BY a) > 0"), "a"));
        assert!(expr_references_column(&check_expr("count(*) OVER (PARTITION BY a) > 0"), "a"));
        assert!(!expr_references_column(&check_expr("count(*) OVER (ORDER BY x) > 0"), "a"));
    }

    #[test]
    fn expr_ref_ignores_unrelated_names_and_literals() {
        assert!(!expr_references_column(&check_expr("b > 0"), "a"));
        // A string literal that spells the column name is not a column reference.
        assert!(!expr_references_column(&check_expr("b <> 'a'"), "a"));
        assert!(!expr_references_column(&check_expr("1 + 2 > 0"), "a"));
    }

    #[test]
    fn expr_ref_treats_any_subquery_as_a_reference_fail_closed() {
        // A nested SELECT (scalar subquery / EXISTS / IN (SELECT …)) is not walked into;
        // it is treated as a reference so a drop cannot slip through a constraint we
        // cannot analyze (the fail-CLOSED rule). Each case below reports the
        // reference PURELY via the subquery arm — the column appears nowhere else in the
        // expression — so a regression back to `false` would flip every assertion.
        assert!(expr_references_column(&check_expr("1 > (SELECT max(z) FROM t)"), "a")); // scalar subquery
        assert!(expr_references_column(&check_expr("EXISTS (SELECT 1 FROM t)"), "a")); // EXISTS
        assert!(expr_references_column(&check_expr("1 IN (SELECT z FROM t)"), "a")); // IN (SELECT)
    }

    #[test]
    fn neighbour_check_referencing_dropped_col_through_varied_exprs_blocks() {
        // A *surviving* neighbour's CHECK reaches the dropped column `b` through a
        // different expression shape in each case; all must block the drop (this is the
        // behaviour the walker's per-variant coverage exists to guarantee).
        for sql in [
            "CREATE TABLE t(a CHECK(NOT b), b, c)",
            "CREATE TABLE t(a CHECK(b ISNULL), b, c)",
            "CREATE TABLE t(a CHECK(b LIKE 'x%'), b, c)",
            "CREATE TABLE t(a CHECK(b COLLATE nocase = 'x'), b, c)",
        ] {
            assert!(block(sql, "b").is_some(), "{sql} should block dropping b");
        }
    }

    // --- create_table_drop_block: allowed -------------------------------------

    #[test]
    fn plain_middle_column_is_droppable() {
        assert_eq!(block("CREATE TABLE t(a, b, c)", "b"), None);
        // A CHECK on the dropped column itself does NOT block (it goes with the column).
        assert_eq!(block("CREATE TABLE t(a, b CHECK(b > 0), c)", "b"), None);
        // A neighbour's CHECK that does not name the column is fine.
        assert_eq!(block("CREATE TABLE t(a CHECK(a > 0), b, c)", "b"), None);
    }

    // --- create_table_drop_block: blocked -------------------------------------

    #[test]
    fn column_level_primary_key_and_unique_block() {
        assert!(block("CREATE TABLE t(a INTEGER PRIMARY KEY, b)", "a").is_some());
        assert!(block("CREATE TABLE t(a UNIQUE, b)", "a").is_some());
    }

    #[test]
    fn table_level_constraints_naming_the_column_block() {
        assert!(block("CREATE TABLE t(a, b, PRIMARY KEY(a, b))", "b").is_some());
        assert!(block("CREATE TABLE t(a, b, UNIQUE(b))", "b").is_some());
        assert!(block("CREATE TABLE t(a, b, CHECK(b > 0))", "b").is_some());
        assert!(
            block("CREATE TABLE t(a, b, FOREIGN KEY(b) REFERENCES u(x))", "b").is_some()
        );
    }

    #[test]
    fn another_columns_check_or_generated_expr_blocks() {
        assert!(block("CREATE TABLE t(a CHECK(a > b), b, c)", "b").is_some());
        assert!(block("CREATE TABLE t(a, b, g AS (b + a))", "b").is_some());
    }

    #[test]
    fn surviving_subquery_constraint_blocks_fail_closed() {
        // The reachable-corruption witness: this engine's parser accepts a subquery inside
        // a CHECK / generated expression, so one can be stored verbatim on page 1. We do
        // not analyze the nested SELECT, so we fail CLOSED — any surviving subquery blocks
        // the drop (real sqlite refuses these too) rather than leaving a dangling reference
        // to the dropped column in a schema that no longer round-trips.
        assert!(
            block("CREATE TABLE t(a, b, CHECK(a > (SELECT max(b) FROM t)))", "b").is_some(),
            "a surviving table-level CHECK subquery must block the drop"
        );
        assert!(
            block("CREATE TABLE t(a CHECK(a > (SELECT max(b) FROM t)), b, c)", "b").is_some(),
            "a surviving column-level CHECK subquery must block the drop"
        );
        assert!(
            block("CREATE TABLE t(a, b, g AS (a + (SELECT count(*) FROM t)))", "b").is_some(),
            "a surviving generated-column subquery must block the drop"
        );
    }

    #[test]
    fn dropped_columns_own_foreign_key_does_not_block() {
        // A column-level FK on the dropped column is removed with it (the reparse model
        // §5.1: nothing surviving references the column), so it is droppable.
        assert_eq!(block("CREATE TABLE t(a, b REFERENCES u(x))", "b"), None);
    }
}
