//! Result-column naming: the label an unaliased SELECT expression gets in the
//! output (`QueryResult.columns`).
//!
//! SQLite names an unaliased result column after the *source text* of its
//! expression (so `SELECT a+1` yields a column literally named `a+1`). The AST here
//! carries no source spans, so [`result_column_name`] reconstructs a close textual
//! form: a bare column keeps its name, and any other expression is rendered back to
//! a canonical SQL-ish string. Whitespace/casing of the original may differ, but a
//! bare column — the overwhelmingly common case and the one tests rely on — is
//! exact.

use minisqlite_sql::{BinaryOp, Expr, FunctionArgs, Literal, UnaryOp};

/// The output name for a result-column expression without an explicit alias.
pub fn result_column_name(e: &Expr) -> String {
    match e {
        // A bare column is named exactly by its column name (not the qualifier).
        Expr::Column { name, .. } => name.clone(),
        // Parentheses/COLLATE are transparent for naming purposes.
        Expr::Parenthesized(list) if list.len() == 1 => result_column_name(&list[0]),
        other => render(other),
    }
}

/// Render an expression back to a canonical SQL-ish string for use as a label.
fn render(e: &Expr) -> String {
    match e {
        Expr::Literal(lit) => render_literal(lit),
        Expr::Column { schema, table, name, .. } => {
            let mut s = String::new();
            if let Some(sc) = schema {
                s.push_str(sc);
                s.push('.');
            }
            if let Some(t) = table {
                s.push_str(t);
                s.push('.');
            }
            s.push_str(name);
            s
        }
        Expr::BindParam(_) => "?".to_string(),
        Expr::Unary { op, expr } => format!("{}{}", render_unary(*op), render(expr)),
        Expr::Binary { op, left, right } => {
            format!("{}{}{}", render(left), render_binary(*op), render(right))
        }
        Expr::Function { name, distinct, args, .. } => {
            let inner = match args {
                FunctionArgs::Star => "*".to_string(),
                FunctionArgs::Empty => String::new(),
                FunctionArgs::List(list) => {
                    list.iter().map(render).collect::<Vec<_>>().join(",")
                }
            };
            let d = if *distinct { "DISTINCT " } else { "" };
            format!("{name}({d}{inner})")
        }
        Expr::Cast { expr, type_name } => format!("CAST({} AS {})", render(expr), type_name),
        Expr::Collate { expr, collation } => format!("{} COLLATE {}", render(expr), collation),
        Expr::Parenthesized(list) => {
            format!("({})", list.iter().map(render).collect::<Vec<_>>().join(","))
        }
        Expr::IsNull(x) => format!("{} ISNULL", render(x)),
        Expr::NotNull(x) => format!("{} NOTNULL", render(x)),
        // Less-common forms: a plausible rendering, not a source reproduction.
        Expr::Between { negated, expr, low, high } => format!(
            "{}{} BETWEEN {} AND {}",
            render(expr),
            if *negated { " NOT" } else { "" },
            render(low),
            render(high)
        ),
        Expr::Like { negated, lhs, .. } => {
            format!("{}{} LIKE ...", render(lhs), if *negated { " NOT" } else { "" })
        }
        Expr::In { negated, expr, .. } => {
            format!("{}{} IN (...)", render(expr), if *negated { " NOT" } else { "" })
        }
        Expr::Case { .. } => "CASE".to_string(),
        Expr::Exists { .. } => "EXISTS (...)".to_string(),
        Expr::Subquery(_) => "(subquery)".to_string(),
        Expr::Raise(_) => "RAISE(...)".to_string(),
    }
}

fn render_literal(lit: &Literal) -> String {
    match lit {
        Literal::Null => "NULL".to_string(),
        Literal::Integer(i) => i.to_string(),
        Literal::Real(r) => r.to_string(),
        Literal::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Literal::Blob(_) => "x'..'".to_string(),
        Literal::True => "TRUE".to_string(),
        Literal::False => "FALSE".to_string(),
        Literal::CurrentDate => "CURRENT_DATE".to_string(),
        Literal::CurrentTime => "CURRENT_TIME".to_string(),
        Literal::CurrentTimestamp => "CURRENT_TIMESTAMP".to_string(),
    }
}

fn render_unary(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Negative => "-",
        UnaryOp::Positive => "+",
        UnaryOp::Not => "NOT ",
        UnaryOp::BitNot => "~",
    }
}

fn render_binary(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Concat => "||",
        BinaryOp::JsonArrow => "->",
        BinaryOp::JsonArrow2 => "->>",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::LShift => "<<",
        BinaryOp::RShift => ">>",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitOr => "|",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::Eq => "=",
        BinaryOp::Ne => "<>",
        BinaryOp::Is => " IS ",
        BinaryOp::IsNot => " IS NOT ",
        BinaryOp::And => " AND ",
        BinaryOp::Or => " OR ",
    }
}
