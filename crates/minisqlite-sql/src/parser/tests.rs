//! Parser tests: expression precedence, the core statements, multi-statement
//! programs, literal/identifier/bind-param lexing, and the loud-failure gaps.
//!
//! Precedence is checked with a compact s-expression rendering (`se`) so a test
//! reads as `input -> fully-parenthesized shape`, which pins associativity and
//! binding power directly against `spec/sqlite-doc/lang_expr.html`.

use crate::ast_ddl::*;
use crate::ast_dml::*;
use crate::ast_expr::*;
use crate::ast_select::*;
use crate::ast_stmt::*;
use crate::parse;
use crate::parser::{parse_expr_str, parse_one};

// --- s-expression rendering of an expression (precedence oracle) -------------

fn op_sym(op: BinaryOp) -> &'static str {
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
        BinaryOp::Ne => "!=",
        BinaryOp::Is => "is",
        BinaryOp::IsNot => "isnot",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
    }
}

fn lit(l: &Literal) -> String {
    match l {
        Literal::Null => "null".into(),
        Literal::Integer(n) => n.to_string(),
        Literal::Real(r) => format!("{r}"),
        Literal::Text(t) => format!("'{t}'"),
        Literal::Blob(b) => format!("blob{}", b.len()),
        Literal::CurrentTime => "CURRENT_TIME".into(),
        Literal::CurrentDate => "CURRENT_DATE".into(),
        Literal::CurrentTimestamp => "CURRENT_TIMESTAMP".into(),
        Literal::True => "true".into(),
        Literal::False => "false".into(),
    }
}

fn join(exprs: &[Expr]) -> String {
    exprs.iter().map(sexpr).collect::<Vec<_>>().join(" ")
}

fn sexpr(e: &Expr) -> String {
    match e {
        Expr::Literal(l) => lit(l),
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
        Expr::BindParam(p) => match p {
            BindParam::Anonymous => "?".into(),
            BindParam::Numbered(n) => format!("?{n}"),
            BindParam::Named(n) => n.clone(),
        },
        Expr::Unary { op, expr } => {
            let o = match op {
                UnaryOp::Negative => "neg",
                UnaryOp::Positive => "pos",
                UnaryOp::Not => "not",
                UnaryOp::BitNot => "bitnot",
            };
            format!("({o} {})", sexpr(expr))
        }
        Expr::Binary { op, left, right } => {
            format!("({} {} {})", op_sym(*op), sexpr(left), sexpr(right))
        }
        Expr::Function { name, distinct, args, .. } => {
            let a = match args {
                FunctionArgs::Star => "*".into(),
                FunctionArgs::Empty => String::new(),
                FunctionArgs::List(l) => join(l),
            };
            let d = if *distinct { "distinct " } else { "" };
            format!("({name} {d}{a})")
        }
        Expr::Cast { expr, type_name } => format!("(cast {} {type_name})", sexpr(expr)),
        Expr::Collate { expr, collation } => format!("(collate {} {collation})", sexpr(expr)),
        Expr::Like { negated, kind, lhs, rhs, escape } => {
            let k = match kind {
                LikeKind::Like => "like",
                LikeKind::Glob => "glob",
                LikeKind::Regexp => "regexp",
                LikeKind::Match => "match",
            };
            let n = if *negated { "not-" } else { "" };
            match escape {
                Some(esc) => format!("({n}{k} {} {} escape {})", sexpr(lhs), sexpr(rhs), sexpr(esc)),
                None => format!("({n}{k} {} {})", sexpr(lhs), sexpr(rhs)),
            }
        }
        Expr::Between { negated, expr, low, high } => {
            let n = if *negated { "not-" } else { "" };
            format!("({n}between {} {} {})", sexpr(expr), sexpr(low), sexpr(high))
        }
        Expr::In { negated, expr, rhs } => {
            let n = if *negated { "not-" } else { "" };
            let r = match rhs {
                InRhs::List(l) => format!("list[{}]", join(l)),
                InRhs::Select(_) => "select".into(),
                InRhs::Table { name, .. } => format!("table:{}", name.name),
            };
            format!("({n}in {} {r})", sexpr(expr))
        }
        Expr::Exists { negated, .. } => {
            let n = if *negated { "not-" } else { "" };
            format!("({n}exists select)")
        }
        Expr::Subquery(_) => "(subquery)".into(),
        Expr::Case { operand, whens, else_expr } => {
            let mut s = String::from("(case");
            if let Some(o) = operand {
                s.push(' ');
                s.push_str(&sexpr(o));
            }
            for (c, r) in whens {
                s.push_str(&format!(" when {} then {}", sexpr(c), sexpr(r)));
            }
            if let Some(e) = else_expr {
                s.push_str(&format!(" else {}", sexpr(e)));
            }
            s.push(')');
            s
        }
        Expr::IsNull(e) => format!("(isnull {})", sexpr(e)),
        Expr::NotNull(e) => format!("(notnull {})", sexpr(e)),
        Expr::Raise(_) => "(raise)".into(),
        Expr::Parenthesized(l) => format!("(row {})", join(l)),
    }
}

/// Parse a standalone expression and render it as an s-expression.
fn se(sql: &str) -> String {
    sexpr(&parse_expr_str(sql).unwrap_or_else(|e| panic!("parse {sql:?}: {e:?}")))
}

fn one(sql: &str) -> Statement {
    parse_one(sql).unwrap_or_else(|e| panic!("parse {sql:?}: {e:?}"))
}

// --- expression precedence ---------------------------------------------------

#[test]
fn precedence_arithmetic() {
    assert_eq!(se("1+2*3"), "(+ 1 (* 2 3))");
    assert_eq!(se("1*2+3"), "(+ (* 1 2) 3)");
    assert_eq!(se("(1+2)*3"), "(* (+ 1 2) 3)");
    assert_eq!(se("10-2-3"), "(- (- 10 2) 3)");
    assert_eq!(se("2*-3"), "(* 2 (neg 3))");
    assert_eq!(se("a/b%c"), "(% (/ a b) c)");
}

#[test]
fn precedence_boolean() {
    assert_eq!(se("a AND b OR c"), "(or (and a b) c)");
    assert_eq!(se("a OR b AND c"), "(or a (and b c))");
    assert_eq!(se("NOT a = b"), "(not (= a b))");
    assert_eq!(se("NOT a AND b"), "(and (not a) b)");
    assert_eq!(se("a = b AND c = d"), "(and (= a b) (= c d))");
}

#[test]
fn precedence_concat_bitwise_and_unary() {
    assert_eq!(se("-a || b"), "(|| (neg a) b)");
    assert_eq!(se("~a + b"), "(+ (bitnot a) b)");
    assert_eq!(se("a || b || c"), "(|| (|| a b) c)");
    // bitwise binds tighter than comparison; comparison tighter than equality.
    assert_eq!(se("a & b = c"), "(= (& a b) c)");
    assert_eq!(se("a = b < c"), "(= a (< b c))");
    assert_eq!(se("a < b = c"), "(= (< a b) c)");
    assert_eq!(se("a = b = c"), "(= (= a b) c)");
}

#[test]
fn precedence_collate() {
    // COLLATE binds tighter than the arithmetic/comparison operators.
    assert_eq!(se("a < b COLLATE NOCASE"), "(< a (collate b NOCASE))");
}

#[test]
fn between_like_in() {
    assert_eq!(se("a BETWEEN 1 AND 2 AND c"), "(and (between a 1 2) c)");
    assert_eq!(se("x NOT BETWEEN 1 AND 2"), "(not-between x 1 2)");
    assert_eq!(se("name LIKE 'a%' ESCAPE '!'"), "(like name 'a%' escape '!')");
    assert_eq!(se("s GLOB '*.rs'"), "(glob s '*.rs')");
    assert_eq!(se("x NOT LIKE 'p'"), "(not-like x 'p')");
    assert_eq!(se("x NOT IN (1, 2, 3)"), "(not-in x list[1 2 3])");
    assert_eq!(se("x IN ()"), "(in x list[])");
    assert_eq!(se("x IN (SELECT id FROM t)"), "(in x select)");
    assert_eq!(se("x IN t"), "(in x table:t)");
}

#[test]
fn is_and_null_forms() {
    assert_eq!(se("a IS NULL"), "(is a null)");
    assert_eq!(se("a IS NOT NULL"), "(isnot a null)");
    assert_eq!(se("a ISNULL"), "(isnull a)");
    assert_eq!(se("a NOTNULL"), "(notnull a)");
    assert_eq!(se("a NOT NULL"), "(notnull a)");
    assert_eq!(se("a IS NOT DISTINCT FROM b"), "(is a b)");
    assert_eq!(se("a IS DISTINCT FROM b"), "(isnot a b)");
}

#[test]
fn functions_case_cast_exists() {
    assert_eq!(se("f(g(x), y)"), "(f (g x) y)");
    assert_eq!(se("count(*)"), "(count *)");
    assert_eq!(se("count(DISTINCT x)"), "(count distinct x)");
    assert_eq!(se("now()"), "(now )");
    assert_eq!(se("CASE WHEN a THEN 1 ELSE 2 END"), "(case when a then 1 else 2)");
    assert_eq!(
        se("CASE x WHEN 1 THEN 'a' WHEN 2 THEN 'b' END"),
        "(case x when 1 then 'a' when 2 then 'b')"
    );
    assert_eq!(se("CAST(x AS INTEGER)"), "(cast x INTEGER)");
    assert_eq!(se("CAST(x AS VARCHAR(10))"), "(cast x VARCHAR(10))");
    assert_eq!(se("EXISTS (SELECT 1)"), "(exists select)");
    assert_eq!(se("NOT EXISTS (SELECT 1)"), "(not (exists select))");
}

#[test]
fn aggregate_order_by_parses_into_function_order_by() {
    // lang_aggfunc.html #aggorderby: an aggregate call may carry an ORDER BY after
    // its last argument. It parses onto `Expr::Function.order_by` (the `se` oracle
    // omits it, so inspect the AST directly), leaving the argument list intact.
    // `Expr` implements `Drop` (iterative teardown), so match by reference.
    let e = parse_expr_str("group_concat(v, ',' ORDER BY v DESC, k)").expect("parse");
    match &e {
        Expr::Function { name, args, order_by, .. } => {
            assert_eq!(name, "group_concat");
            match args {
                FunctionArgs::List(l) => assert_eq!(l.len(), 2, "the two call args stay in the list"),
                other => panic!("expected an argument list, got {other:?}"),
            }
            assert_eq!(order_by.len(), 2, "two ORDER BY terms");
            assert!(matches!(order_by[0].order, Some(SortOrder::Desc)), "first term is DESC");
            assert!(order_by[1].order.is_none(), "second term has no explicit direction");
        }
        other => panic!("expected a function call, got {other:?}"),
    }
}

#[test]
fn function_without_order_by_has_empty_order_by() {
    // The common no-ORDER-BY call leaves `order_by` empty (not a phantom term).
    let e = parse_expr_str("count(x)").expect("parse");
    match &e {
        Expr::Function { order_by, .. } => assert!(order_by.is_empty()),
        other => panic!("expected a function call, got {other:?}"),
    }
}

#[test]
fn columns_params_literals() {
    assert_eq!(se("a"), "a");
    assert_eq!(se("t.a"), "t.a");
    assert_eq!(se("s.t.a"), "s.t.a");
    assert_eq!(se("?"), "?");
    assert_eq!(se("?5"), "?5");
    assert_eq!(se(":name"), ":name");
    assert_eq!(se("@x"), "@x");
    assert_eq!(se("$y"), "$y");
    assert_eq!(se("TRUE"), "true");
    assert_eq!(se("FALSE"), "false");
    assert_eq!(se("NULL"), "null");
    assert_eq!(se("CURRENT_TIMESTAMP"), "CURRENT_TIMESTAMP");
    assert_eq!(se("(1, 2, 3)"), "(row 1 2 3)");
    assert_eq!(se("(a)"), "a");
}

// --- literal / identifier lexing (through the parser) ------------------------

fn first_col_expr(sql: &str) -> Expr {
    // Match by reference and clone: `SelectBody`/`Expr` implement `Drop` (iterative
    // teardown), so their fields cannot be moved out by pattern.
    let Statement::Select(sel) = one(sql) else { panic!("not a SELECT: {sql:?}") };
    let SelectBody::Select(SelectCore::Query { columns, .. }) = &sel.body else {
        panic!("not a query core: {sql:?}")
    };
    match columns.first().unwrap() {
        ResultColumn::Expr { expr, .. } => expr.clone(),
        other => panic!("not an expression column: {other:?}"),
    }
}

#[test]
fn lexing_numbers_strings_blobs() {
    assert_eq!(first_col_expr("SELECT 42"), Expr::Literal(Literal::Integer(42)));
    assert_eq!(first_col_expr("SELECT 0x10"), Expr::Literal(Literal::Integer(16)));
    assert_eq!(first_col_expr("SELECT 1.5e3"), Expr::Literal(Literal::Real(1500.0)));
    assert_eq!(first_col_expr("SELECT .25"), Expr::Literal(Literal::Real(0.25)));
    assert_eq!(first_col_expr("SELECT 'it''s'"), Expr::Literal(Literal::Text("it's".into())));
    assert_eq!(
        first_col_expr("SELECT x'48656c6c6f'"),
        Expr::Literal(Literal::Blob(b"Hello".to_vec()))
    );
}

#[test]
fn lexing_quoted_identifiers() {
    // A bracketed / backtick identifier is a column name, not a string.
    assert_eq!(
        first_col_expr("SELECT [order]"),
        Expr::Column { schema: None, table: None, name: "order".into(), from_dqs: false }
    );
    assert_eq!(
        first_col_expr("SELECT `weird name`"),
        Expr::Column { schema: None, table: None, name: "weird name".into(), from_dqs: false }
    );
}

#[test]
fn comments_are_skipped() {
    let e = first_col_expr("SELECT /* block */ 1 -- trailing\n");
    assert_eq!(e, Expr::Literal(Literal::Integer(1)));
}

// --- CREATE TABLE ------------------------------------------------------------

#[test]
fn create_table_columns_and_constraints() {
    let Statement::CreateTable(ct) =
        one("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, UNIQUE(name))")
    else {
        panic!()
    };
    assert_eq!(ct.name.name, "t");
    assert!(!ct.temp && !ct.if_not_exists);
    let CreateTableBody::Columns { columns, constraints, options } = &ct.body else { panic!() };
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[0].name, "id");
    assert_eq!(columns[0].type_name.as_deref(), Some("INTEGER"));
    assert!(matches!(
        columns[0].constraints[0].kind,
        ColumnConstraintKind::PrimaryKey { autoincrement: true, .. }
    ));
    assert_eq!(columns[1].name, "name");
    assert!(matches!(columns[1].constraints[0].kind, ColumnConstraintKind::NotNull { .. }));
    assert_eq!(constraints.len(), 1);
    assert!(matches!(constraints[0].kind, TableConstraintKind::Unique { .. }));
    assert!(!options.without_rowid && !options.strict);
}

#[test]
fn create_table_temp_if_not_exists() {
    let Statement::CreateTable(ct) = one("CREATE TEMP TABLE IF NOT EXISTS x (a)") else { panic!() };
    assert!(ct.temp);
    assert!(ct.if_not_exists);
}

#[test]
fn create_table_options_and_fk() {
    let Statement::CreateTable(ct) = one(
        "CREATE TABLE t (a INTEGER, b, FOREIGN KEY (a) REFERENCES u(id) ON DELETE CASCADE) WITHOUT ROWID, STRICT",
    ) else {
        panic!()
    };
    let CreateTableBody::Columns { constraints, options, .. } = &ct.body else { panic!() };
    assert!(options.without_rowid && options.strict);
    let TableConstraintKind::ForeignKey { clause, .. } = &constraints[0].kind else { panic!() };
    assert_eq!(clause.table, "u");
    assert_eq!(clause.actions.len(), 1);
    assert!(matches!(
        clause.actions[0],
        ForeignKeyAction::OnDelete(ReferentialAction::Cascade)
    ));
}

#[test]
fn create_table_as_select() {
    let Statement::CreateTable(ct) = one("CREATE TABLE t AS SELECT * FROM u") else { panic!() };
    assert!(matches!(ct.body, CreateTableBody::AsSelect(_)));
}

#[test]
fn create_table_default_and_check() {
    let Statement::CreateTable(ct) =
        one("CREATE TABLE t (a INT DEFAULT 5, b TEXT DEFAULT 'x', c CHECK (c > 0))")
    else {
        panic!()
    };
    let CreateTableBody::Columns { columns, .. } = &ct.body else { panic!() };
    assert!(matches!(
        columns[0].constraints[0].kind,
        ColumnConstraintKind::Default(DefaultValue::Literal(Literal::Integer(5)))
    ));
    assert!(matches!(columns[2].constraints[0].kind, ColumnConstraintKind::Check(_)));
}

// --- CREATE INDEX ------------------------------------------------------------

#[test]
fn create_index_partial() {
    let Statement::CreateIndex(ci) = one("CREATE UNIQUE INDEX i ON t (a, b DESC) WHERE a > 0")
    else {
        panic!()
    };
    assert!(ci.unique);
    assert_eq!(ci.table, "t");
    assert_eq!(ci.columns.len(), 2);
    assert!(matches!(ci.columns[1].order, Some(SortOrder::Desc)));
    assert!(ci.where_clause.is_some());
}

// --- INSERT ------------------------------------------------------------------

#[test]
fn insert_values_multirow() {
    let Statement::Insert(ins) = one("INSERT INTO t (a, b) VALUES (1, 2), (3, 4)") else {
        panic!()
    };
    assert_eq!(ins.table.name, "t");
    assert_eq!(ins.columns.as_ref().unwrap(), &["a".to_string(), "b".to_string()]);
    let InsertSource::Values(rows) = &ins.source else { panic!() };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), 2);
}

#[test]
fn insert_select_and_default_values() {
    let Statement::Insert(ins) = one("INSERT INTO t SELECT * FROM u") else { panic!() };
    assert!(matches!(ins.source, InsertSource::Select(_)));

    let Statement::Insert(ins) = one("INSERT INTO t DEFAULT VALUES") else { panic!() };
    assert!(matches!(ins.source, InsertSource::DefaultValues));
}

#[test]
fn insert_or_replace_forms() {
    let Statement::Insert(ins) = one("INSERT OR REPLACE INTO t VALUES (1)") else { panic!() };
    assert_eq!(ins.or_conflict, Some(ConflictClause::Replace));

    let Statement::Insert(ins) = one("REPLACE INTO t VALUES (1)") else { panic!() };
    assert_eq!(ins.or_conflict, Some(ConflictClause::Replace));

    let Statement::Insert(ins) = one("INSERT OR IGNORE INTO t VALUES (1)") else { panic!() };
    assert_eq!(ins.or_conflict, Some(ConflictClause::Ignore));
}

#[test]
fn insert_upsert_and_returning() {
    let Statement::Insert(ins) =
        one("INSERT INTO t (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET n = n + 1 RETURNING *")
    else {
        panic!()
    };
    assert_eq!(ins.upsert.len(), 1);
    assert!(ins.upsert[0].target.is_some());
    assert!(matches!(ins.upsert[0].action, UpsertAction::Update { .. }));
    assert_eq!(ins.returning.len(), 1);

    let Statement::Insert(ins) = one("INSERT INTO t VALUES (1) ON CONFLICT DO NOTHING") else {
        panic!()
    };
    assert_eq!(ins.upsert.len(), 1);
    assert!(ins.upsert[0].target.is_none());
    assert!(matches!(ins.upsert[0].action, UpsertAction::Nothing));
}

// --- SELECT ------------------------------------------------------------------

#[test]
fn select_full_shape() {
    let Statement::Select(sel) = one(
        "SELECT DISTINCT a, b AS c, x.* FROM x JOIN y ON x.id = y.id \
         WHERE a > 1 GROUP BY a HAVING count(*) > 1 ORDER BY a DESC LIMIT 10 OFFSET 5",
    ) else {
        panic!()
    };
    assert_eq!(sel.order_by.len(), 1);
    assert!(matches!(sel.order_by[0].order, Some(SortOrder::Desc)));
    let limit = sel.limit.as_ref().unwrap();
    assert!(limit.offset.is_some());
    let SelectBody::Select(SelectCore::Query {
        distinct,
        columns,
        from,
        where_clause,
        group_by,
        having,
        ..
    }) = &sel.body
    else {
        panic!()
    };
    assert_eq!(*distinct, Distinct::Distinct);
    assert_eq!(columns.len(), 3);
    assert!(matches!(&columns[1], ResultColumn::Expr { alias: Some(a), .. } if a == "c"));
    assert!(matches!(&columns[2], ResultColumn::TableStar(t) if t == "x"));
    assert!(matches!(from.as_ref().unwrap(), JoinTree::Join { .. }));
    assert!(where_clause.is_some());
    assert_eq!(group_by.len(), 1);
    assert!(having.is_some());
}

#[test]
fn select_join_variants() {
    let Statement::Select(sel) =
        one("SELECT * FROM a LEFT JOIN b USING (id), c CROSS JOIN d ON c.x = d.x")
    else {
        panic!()
    };
    let SelectBody::Select(SelectCore::Query { from: Some(tree), .. }) = &sel.body else {
        panic!()
    };
    // Outer node is the CROSS JOIN with an ON constraint.
    let JoinTree::Join { op, constraint, .. } = tree else { panic!() };
    assert_eq!(op.kind, JoinKind::Cross);
    assert!(matches!(constraint, Some(JoinConstraint::On(_))));
}

#[test]
fn select_compound_union() {
    let Statement::Select(sel) = one("SELECT 1 UNION SELECT 2 UNION ALL SELECT 3") else {
        panic!()
    };
    let SelectBody::Compound { op, left, .. } = &sel.body else { panic!() };
    assert_eq!(*op, CompoundOp::UnionAll);
    assert!(matches!(**left, SelectBody::Compound { op: CompoundOp::Union, .. }));
}

#[test]
fn join_and_window_keywords_are_usable_as_names() {
    // SQLite's tokenizer reclassifies the JOIN_KW / OVER / WINDOW keywords (and FILTER,
    // by look-ahead) to identifiers in name positions, so all of these are legal column
    // names, table names, and `AS` aliases. Each parses without error.
    assert!(parse("CREATE TABLE node(id, left, right, full, inner, outer, cross, natural)").is_ok());
    assert!(parse("CREATE TABLE settings(over, filter, window)").is_ok());
    assert!(parse("CREATE TABLE left(x)").is_ok());
    assert!(parse("SELECT left, right FROM node").is_ok());
    assert!(parse("SELECT node.full FROM node").is_ok());
    assert!(parse("SELECT 1 AS left, 2 AS window").is_ok());
    assert!(parse("SELECT left.* FROM left").is_ok());
    assert!(parse("UPDATE node SET left = 1 WHERE right = 2").is_ok());
}

#[test]
fn join_keywords_as_names_do_not_break_join_or_window_parsing() {
    // The flip above must not swallow a join operator or a WINDOW clause as an alias.
    // `LEFT`/`RIGHT`/etc. in the operator slot still parse as joins:
    let Statement::Select(sel) = one("SELECT a.id FROM a LEFT JOIN b ON a.id = b.id") else {
        panic!()
    };
    let SelectBody::Select(SelectCore::Query { from: Some(tree), .. }) = &sel.body else {
        panic!()
    };
    let JoinTree::Join { op, .. } = tree else { panic!("expected a join") };
    assert_eq!(op.kind, JoinKind::Left);
    // A trailing WINDOW clause is not eaten as `t`'s alias:
    assert!(
        parse("SELECT row_number() OVER w FROM t WINDOW w AS (ORDER BY x)").is_ok(),
        "WINDOW clause after a table must still parse"
    );
    // Aliased tables around a join still work.
    assert!(parse("SELECT l.id FROM a l LEFT JOIN b r ON l.id = r.id").is_ok());
}

#[test]
fn select_values_form() {
    let Statement::Select(sel) = one("VALUES (1, 2), (3, 4)") else { panic!() };
    let SelectBody::Select(SelectCore::Values(rows)) = &sel.body else { panic!() };
    assert_eq!(rows.len(), 2);
}

#[test]
fn select_with_cte() {
    let Statement::Select(sel) =
        one("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT * FROM c")
    else {
        panic!()
    };
    let with = sel.with.as_ref().unwrap();
    assert!(with.recursive);
    assert_eq!(with.ctes.len(), 1);
    assert_eq!(with.ctes[0].name, "c");
    assert_eq!(with.ctes[0].columns.as_ref().unwrap(), &["n".to_string()]);
}

#[test]
fn select_order_by_nulls_and_collate() {
    let Statement::Select(sel) =
        one("SELECT a FROM t ORDER BY a COLLATE NOCASE ASC NULLS LAST, b DESC")
    else {
        panic!()
    };
    assert_eq!(sel.order_by.len(), 2);
    // COLLATE is an expression operator, so `a COLLATE NOCASE` folds into `expr`;
    // OrderingTerm.collation stays None (see `parse_ordering_term`).
    assert!(matches!(
        &sel.order_by[0].expr,
        Expr::Collate { collation, .. } if collation == "NOCASE"
    ));
    assert_eq!(sel.order_by[0].collation, None);
    assert!(matches!(sel.order_by[0].order, Some(SortOrder::Asc)));
    assert!(matches!(sel.order_by[0].nulls, Some(NullsOrder::Last)));
    assert!(matches!(sel.order_by[1].order, Some(SortOrder::Desc)));
}

#[test]
fn select_limit_comma_form() {
    // `LIMIT offset, count` is sugar for `LIMIT count OFFSET offset`.
    let Statement::Select(sel) = one("SELECT a FROM t LIMIT 5, 10") else { panic!() };
    let limit = sel.limit.unwrap();
    assert_eq!(limit.limit, Expr::Literal(Literal::Integer(10)));
    assert_eq!(limit.offset, Some(Expr::Literal(Literal::Integer(5))));
}

// --- UPDATE / DELETE ---------------------------------------------------------

#[test]
fn update_basic_and_tuple_set() {
    let Statement::Update(u) = one("UPDATE t SET a = 1, b = 2 WHERE id = 3") else { panic!() };
    assert_eq!(u.set.len(), 2);
    assert!(matches!(&u.set[0], SetClause::Column { name, .. } if name == "a"));
    assert!(u.where_clause.is_some());

    let Statement::Update(u) = one("UPDATE t SET (a, b) = (1, 2) FROM u WHERE t.id = u.id") else {
        panic!()
    };
    assert!(matches!(&u.set[0], SetClause::Columns { names, .. } if names.len() == 2));
    assert!(u.from.is_some());
}

#[test]
fn update_or_conflict() {
    let Statement::Update(u) = one("UPDATE OR ROLLBACK t SET a = 1") else { panic!() };
    assert_eq!(u.or_conflict, Some(ConflictClause::Rollback));
}

#[test]
fn delete_with_returning() {
    let Statement::Delete(d) = one("DELETE FROM t WHERE a = 1 RETURNING id") else { panic!() };
    assert!(d.where_clause.is_some());
    assert_eq!(d.returning.len(), 1);
}

// --- DROP / transactions / EXPLAIN ------------------------------------------

#[test]
fn drop_statements() {
    let Statement::Drop(d) = one("DROP TABLE IF EXISTS t") else { panic!() };
    assert_eq!(d.kind, DropKind::Table);
    assert!(d.if_exists);

    let Statement::Drop(d) = one("DROP INDEX i") else { panic!() };
    assert_eq!(d.kind, DropKind::Index);
    assert!(!d.if_exists);
}

#[test]
fn transaction_control() {
    assert!(matches!(one("BEGIN"), Statement::Begin { mode: TransactionMode::Deferred }));
    assert!(matches!(
        one("BEGIN IMMEDIATE TRANSACTION"),
        Statement::Begin { mode: TransactionMode::Immediate }
    ));
    assert!(matches!(one("COMMIT"), Statement::Commit));
    assert!(matches!(one("END TRANSACTION"), Statement::Commit));
    assert!(matches!(one("ROLLBACK"), Statement::Rollback { to_savepoint: None }));
    let Statement::Rollback { to_savepoint } = one("ROLLBACK TO SAVEPOINT sp") else { panic!() };
    assert_eq!(to_savepoint.as_deref(), Some("sp"));
    assert!(matches!(one("SAVEPOINT sp"), Statement::Savepoint(s) if s == "sp"));
    assert!(matches!(one("RELEASE sp"), Statement::Release(s) if s == "sp"));
}

#[test]
fn explain_prefixes() {
    assert!(matches!(one("EXPLAIN SELECT 1"), Statement::Explain(_)));
    assert!(matches!(one("EXPLAIN QUERY PLAN SELECT 1"), Statement::ExplainQueryPlan(_)));
}

// --- program-level: multi-statement, empties, trailing separator -------------

#[test]
fn multi_statement_program() {
    let ast = parse("SELECT 1; SELECT 2; ; INSERT INTO t VALUES (1);").expect("parse program");
    assert_eq!(ast.statements.len(), 3);
    assert!(matches!(ast.statements[0], Statement::Select(_)));
    assert!(matches!(ast.statements[2], Statement::Insert(_)));
}

#[test]
fn empty_program_is_empty() {
    assert_eq!(parse("   ;;  -- just a comment\n").unwrap().statements.len(), 0);
    assert_eq!(parse("").unwrap().statements.len(), 0);
}

// --- loud failure on the long tail ------------------------------------------

#[test]
fn unsupported_gaps_error_loudly() {
    // CREATE VIRTUAL TABLE is a defined-but-unparsed production.
    let err = parse("CREATE VIRTUAL TABLE t USING fts5(x)").unwrap_err();
    assert!(format!("{err:?}").contains("unsupported"), "got {err:?}");
}

#[test]
fn precedence_json_arrow_operators() {
    // `->` / `->>` fold to their own BinaryOp variants (json1.html §4.10).
    assert_eq!(se("a -> b"), "(-> a b)");
    assert_eq!(se("a ->> b"), "(->> a b)");
    // Left-associative, and mixed forms nest to the left.
    assert_eq!(se("a -> b -> c"), "(-> (-> a b) c)");
    assert_eq!(se("a -> b ->> c"), "(->> (-> a b) c)");
    // Same binding power as `||` (they share PREC_CONCAT), so they nest left with it.
    assert_eq!(se("a -> b || c"), "(|| (-> a b) c)");
    // `->` binds TIGHTER than `*` (PREC_CONCAT > PREC_MUL), matching SQLite's table.
    assert_eq!(se("a -> b * c"), "(* (-> a b) c)");
    // The verbatim doc chain `... -> 'c' -> 2 ->> 'f'` associates left-to-right.
    assert_eq!(se("j -> 'c' -> 2 ->> 'f'"), "(->> (-> (-> j 'c') 2) 'f')");
}

#[test]
fn syntax_errors_are_reported() {
    assert!(parse("SELECT FROM").is_err());
    assert!(parse("INSERT INTO").is_err());
    assert!(parse("SELECT 1 2 3").is_err());
}

// --- recursion-depth guard (no native stack overflow on hostile input) -------
//
// A recursive-descent parser overflows the native call stack on deeply nested
// input; `MAX_PARSE_DEPTH` bounds that so the parser returns an error instead of
// aborting the process (as SQLite does — SQLITE_MAX_EXPR_DEPTH). These tests run
// on the default test-thread stack ON PURPOSE: if the guard regresses to
// unbounded recursion the process aborts with SIGABRT (a stack overflow) rather
// than the assertion merely failing — the abort *is* the regression signal.

#[test]
fn deeply_nested_input_errors_instead_of_overflowing_the_stack() {
    // Every distinct recursion head must be bounded, so exercise each one with a
    // nesting well beyond MAX_PARSE_DEPTH (a few hundred already trips the guard;
    // no need for pathologically huge inputs).
    let deep_parens = format!("SELECT {}1{}", "(".repeat(400), ")".repeat(400));
    assert!(parse(&deep_parens).is_err(), "deep parenthesized expression");

    let mut subq = String::from("SELECT 1");
    for _ in 0..400 {
        subq = format!("SELECT * FROM ({subq})");
    }
    assert!(parse(&subq).is_err(), "deeply nested subqueries");

    let deep_not = format!("SELECT {}1", "NOT ".repeat(400));
    assert!(parse(&deep_not).is_err(), "deep unary NOT chain");

    let deep_from = format!("SELECT * FROM {}t{}", "(".repeat(400), ")".repeat(400));
    assert!(parse(&deep_from).is_err(), "deeply nested parenthesized joins");

    let deep_explain = format!("{}SELECT 1", "EXPLAIN ".repeat(400));
    assert!(parse(&deep_explain).is_err(), "repeated EXPLAIN prefix");
}

#[test]
fn moderate_nesting_and_wide_lists_are_accepted() {
    // Nesting comfortably under the limit parses fine: the guard rejects only
    // pathological depth, never legitimately-structured SQL.
    let parens = format!("SELECT {}1{}", "(".repeat(30), ")".repeat(30));
    assert!(parse(&parens).is_ok(), "30 levels of parens is well under the limit");

    let mut subq = String::from("SELECT 1");
    for _ in 0..15 {
        subq = format!("SELECT * FROM ({subq})");
    }
    assert!(parse(&subq).is_ok(), "15 nested subqueries is under the limit");

    // Width is not depth: a large *sibling* list recurses one level at a time, so
    // the depth guard (which decrements on unwind) must not reject it.
    let wide_values: Vec<&str> = vec!["1"; 5000];
    let wide_in = format!("SELECT 1 WHERE 1 IN ({})", wide_values.join(","));
    assert!(parse(&wide_in).is_ok(), "5000-element IN list is width, not depth");

    let wide_cols = format!("SELECT {}", vec!["1"; 2000].join(","));
    assert!(parse(&wide_cols).is_ok(), "2000-column SELECT is width, not depth");

    let wide_rows: Vec<&str> = vec!["(1)"; 2000];
    let wide_values_stmt = format!("VALUES {}", wide_rows.join(","));
    assert!(parse(&wide_values_stmt).is_ok(), "2000-row VALUES is width, not depth");
}

// --- compound SELECT term limit (matches SQLITE_MAX_COMPOUND_SELECT) ---------
//
// A compound body is a left-nested tree built by a loop, so its depth tracks
// input *width*, not parse recursion — the depth guard does not bound it. SQLite
// caps the term count at 500; matching that both matches real SQLite (a
// longer chain errors, as in real SQLite) and keeps the tree shallow enough that
// its recursive Drop cannot overflow the stack.

#[test]
fn compound_select_term_limit_matches_sqlite() {
    // A normal small compound is unaffected (regression guard).
    assert!(parse("SELECT 1 UNION SELECT 2 UNION ALL SELECT 3").is_ok());

    // 500 terms (499 UNIONs) sits exactly at SQLITE_MAX_COMPOUND_SELECT and parses.
    let at_limit = vec!["SELECT 1"; 500].join(" UNION ");
    assert!(parse(&at_limit).is_ok(), "500-term compound is at the limit");

    // 501 terms exceeds it: SQLite errors ("too many terms in compound SELECT"),
    // so we must too — and failing fast means the oversized tree is never built.
    let over_limit = vec!["SELECT 1"; 501].join(" UNION ");
    let err = parse(&over_limit).unwrap_err();
    assert!(
        format!("{err:?}").contains("too many terms in compound SELECT"),
        "got {err:?}"
    );
}

// --- the other two loop-built, left-nested chains (join list, expr fold) ------
//
// Same shape and hazard as the compound body above: a `Box`-recursive AST node
// built by a *loop*, so its height tracks input WIDTH and the parse-recursion
// depth guard never sees it. Each is bounded at its SQLite limit (64 tables,
// SQLITE_MAX_EXPR_DEPTH=1000) so an oversized chain errors like SQLite; the
// iterative `Drop` tests below give the crash-safety half of the guarantee.

#[test]
fn join_table_limit_matches_sqlite() {
    // A normal join is unaffected (regression guard).
    assert!(parse("SELECT * FROM a, b, c JOIN d").is_ok());

    // 64 tables sits exactly at SQLite's fixed limit and parses.
    let at_limit = (0..64).map(|i| format!("t{i}")).collect::<Vec<_>>().join(",");
    assert!(parse(&format!("SELECT * FROM {at_limit}")).is_ok(), "64 tables is at the limit");

    // 65 tables exceeds it: SQLite errors ("at most 64 tables in a join"), so must we.
    let over_limit = (0..65).map(|i| format!("t{i}")).collect::<Vec<_>>().join(",");
    let err = parse(&format!("SELECT * FROM {over_limit}")).unwrap_err();
    assert!(format!("{err:?}").contains("at most 64 tables in a join"), "got {err:?}");

    // The explicit-JOIN branch shares the same cap via the same helper; exercise its
    // boundary too (a comma-only test wouldn't catch a JOIN-branch-only off-by-one).
    let join_at = (0..64).map(|i| format!("t{i}")).collect::<Vec<_>>().join(" JOIN ");
    assert!(parse(&format!("SELECT * FROM {join_at}")).is_ok(), "64 JOINed tables at limit");
    let join_over = (0..65).map(|i| format!("t{i}")).collect::<Vec<_>>().join(" JOIN ");
    let err = parse(&format!("SELECT * FROM {join_over}")).unwrap_err();
    assert!(format!("{err:?}").contains("at most 64 tables in a join"), "got {err:?}");
}

#[test]
fn expr_depth_limit_matches_sqlite() {
    // A normal expression is unaffected (regression guard).
    assert!(parse("SELECT 1 + 2 * 3 - 4").is_ok());

    // A flat left-associative fold is loop-built, so the depth guard never fires;
    // it is bounded by the expression-height cap instead. 1000 folds is at the
    // limit and parses; 1001 exceeds it and errors like SQLite. Checked on both a
    // generic binary operator (`+`) and the dedicated AND branch.
    let at_limit = format!("SELECT 1{}", " + 1".repeat(1000));
    assert!(parse(&at_limit).is_ok(), "1000-deep fold is at the limit");

    let over_limit = format!("SELECT 1{}", " + 1".repeat(1001));
    let err = parse(&over_limit).unwrap_err();
    assert!(format!("{err:?}").contains("expression tree is too large"), "got {err:?}");

    let over_and = format!("SELECT 1 WHERE 1{}", " AND 1".repeat(1001));
    let err = parse(&over_and).unwrap_err();
    assert!(
        format!("{err:?}").contains("expression tree is too large"),
        "1001-deep AND fold: got {err:?}"
    );
}

// --- iterative Drop (no native stack overflow tearing down a deep AST) --------
//
// The width caps above keep the *parser* from building an over-tall chain, but
// the recursive AST types must also be safe to DROP when a deep tree arrives some
// other way (composed through nesting, or hand-built by a downstream consumer).
// `Expr`, `SelectBody`, and `JoinTree` therefore tear down iteratively. These
// tests hand-build each spine far past any recursive-drop depth and drop it; if a
// type regresses to the derived recursive `Drop`, the process aborts with a stack
// overflow (the abort IS the regression signal), like the depth-guard tests.

#[test]
fn deep_expr_chain_drops_without_overflowing_the_stack() {
    let mut e = Expr::Literal(Literal::Integer(0));
    for _ in 0..200_000 {
        e = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(e),
            right: Box::new(Expr::Literal(Literal::Integer(1))),
        };
    }
    drop(e);
}

#[test]
fn deep_compound_and_join_trees_drop_without_overflowing_the_stack() {
    let mut body = SelectBody::Select(SelectCore::Values(Vec::new()));
    for _ in 0..100_000 {
        body = SelectBody::Compound {
            op: CompoundOp::Union,
            left: Box::new(body),
            right: SelectCore::Values(Vec::new()),
        };
    }
    drop(body);

    let leaf = || TableOrSubquery::Table {
        name: QualifiedName { schema: None, name: String::from("t") },
        alias: None,
        indexed: None,
    };
    let mut tree = JoinTree::Table(leaf());
    for _ in 0..100_000 {
        tree = JoinTree::Join {
            left: Box::new(tree),
            op: JoinOperator { natural: false, kind: JoinKind::Comma },
            right: leaf(),
            constraint: None,
        };
    }
    drop(tree);
}

// --- signed i64::MIN literal folds instead of panicking ----------------------
//
// `0x8000000000000000` tokenizes to i64::MIN (u64 as i64), so a naive `-v` on it
// panics ("attempt to negate with overflow") in debug builds. The DEFAULT and
// PRAGMA value paths must wrap instead, folding `-0x8000000000000000` to i64::MIN
// exactly as SQLite does with its two's-complement negation of the same literal.

#[test]
fn signed_min_hex_literal_folds_instead_of_panicking() {
    let Statement::CreateTable(ct) = one("CREATE TABLE t(x DEFAULT -0x8000000000000000)") else {
        panic!()
    };
    let CreateTableBody::Columns { columns, .. } = &ct.body else { panic!() };
    assert!(matches!(
        columns[0].constraints[0].kind,
        ColumnConstraintKind::Default(DefaultValue::Literal(Literal::Integer(i64::MIN)))
    ));

    let Statement::Pragma {
        arg: Some(PragmaArg::Equals(PragmaValue::Literal(Literal::Integer(v)))),
        ..
    } = one("PRAGMA p = -0x8000000000000000")
    else {
        panic!()
    };
    assert_eq!(v, i64::MIN);
}

// --- per-statement verbatim source (sqlite_schema.sql text) ------------------
//
// `Ast.statement_sources[i]` is the exact source of `statements[i]`: the span from the
// first token to the last token, so surrounding whitespace/comments and the terminating
// `;` are excluded while internal bytes are preserved verbatim. Because the span is
// computed over TOKENS, a `;` inside a literal/identifier/comment never splits.

#[test]
fn statement_sources_align_and_preserve_internal_whitespace() {
    // The canonical example: preserved double space and embedded newline, no leading
    // indentation, no trailing spaces before `;`, no terminating `;`.
    let ast = parse("  CREATE   TABLE t(a INTEGER,\n b TEXT)  ; SELECT 1;").expect("parse");
    assert_eq!(ast.statements.len(), 2);
    assert_eq!(ast.statement_sources.len(), ast.statements.len());
    assert_eq!(ast.statement_sources[0], "CREATE   TABLE t(a INTEGER,\n b TEXT)");
    assert_eq!(ast.statement_sources[1], "SELECT 1");
}

#[test]
fn statement_source_excludes_terminator_and_surrounding_trivia() {
    // A leading comment + whitespace and a trailing comment are both outside the span;
    // it is exactly the statement's own tokens. The `;` inside the trailing comment is
    // not a separator.
    let ast = parse("/* lead */  SELECT 1  /* trail; */ ; SELECT 2").expect("parse");
    assert_eq!(ast.statement_sources, vec!["SELECT 1".to_string(), "SELECT 2".to_string()]);
}

#[test]
fn semicolon_in_string_literal_keeps_insert_one_statement() {
    // Spec example: the `;` inside 'a;b' does not end the INSERT.
    let ast = parse("INSERT INTO t VALUES ('a;b')").expect("parse");
    assert_eq!(ast.statements.len(), 1);
    assert!(matches!(ast.statements[0], Statement::Insert(_)));
    assert_eq!(ast.statement_sources[0], "INSERT INTO t VALUES ('a;b')");
}

#[test]
fn semicolon_in_strings_quoted_idents_and_blobs_does_not_split() {
    // A `;` inside a single-quoted string or any of the three quoted-identifier forms
    // ("double", [bracket], `backtick`) is part of that one token, never a separator;
    // a blob literal is likewise a single token spanned verbatim. So this is ONE
    // statement whose source is the whole input.
    let sql = "SELECT 'a;b', \"c;d\", [e;f], `g;h`, x'3b'";
    let ast = parse(sql).expect("parse");
    assert_eq!(ast.statements.len(), 1);
    assert_eq!(ast.statement_sources[0], sql);
}

#[test]
fn semicolon_in_comments_does_not_split_and_internal_comment_preserved() {
    // Block comment between two tokens keeps its `;` and is preserved verbatim in span.
    let ast = parse("SELECT /* x; y */ 1; SELECT 2").expect("parse");
    assert_eq!(
        ast.statement_sources,
        vec!["SELECT /* x; y */ 1".to_string(), "SELECT 2".to_string()]
    );

    // A line comment (with a `;`) BETWEEN two tokens of one statement is preserved
    // verbatim, newline and all. (A line-comment `;` is never a separator; a real `;`
    // is still required to split statements.)
    let ast = parse("SELECT 1 + -- z; z\n 2").expect("parse");
    assert_eq!(ast.statements.len(), 1);
    assert_eq!(ast.statement_sources[0], "SELECT 1 + -- z; z\n 2");
}

#[test]
fn create_table_source_matches_sqlite_schema_text() {
    // Exactly the bytes SQLite records in sqlite_schema.sql: no leading indent, no
    // trailing `;`, internal layout byte-for-byte (including the quoted name).
    let sql = "CREATE TABLE \"my tbl\"(\n  id INTEGER PRIMARY KEY,\n  v TEXT\n);";
    let ast = parse(sql).expect("parse");
    assert!(matches!(ast.statements[0], Statement::CreateTable(_)));
    assert_eq!(
        ast.statement_sources[0],
        "CREATE TABLE \"my tbl\"(\n  id INTEGER PRIMARY KEY,\n  v TEXT\n)"
    );
}

#[test]
fn create_index_source_is_verbatim() {
    let sql = "CREATE UNIQUE INDEX i ON t (a, b DESC) WHERE a > 0";
    let ast = parse(sql).expect("parse");
    assert!(matches!(ast.statements[0], Statement::CreateIndex(_)));
    assert_eq!(ast.statement_sources[0], sql);
}

#[test]
fn empty_statements_and_trailing_separator_stay_aligned() {
    // Leading `;;`, internal `;;;`, and a trailing `;` produce no spurious entries.
    let ast = parse(";; SELECT 1 ;;; SELECT 2 ;").expect("parse");
    assert_eq!(ast.statement_sources, vec!["SELECT 1".to_string(), "SELECT 2".to_string()]);

    // A program of only separators/whitespace/comments yields empty statements AND
    // empty sources (the invariant holds at zero).
    let ast = parse("   ;;  -- just a comment\n").expect("parse");
    assert!(ast.statements.is_empty());
    assert!(ast.statement_sources.is_empty());
}

#[test]
fn create_trigger_body_semicolons_do_not_split() {
    // A trigger's BEGIN ... END body contains real `;` tokens that `parse_statement`
    // consumes internally (through END, see `parse_create_trigger`), so they never reach
    // parse_program's separator loop: the whole trigger is ONE statement with its internal
    // `;` preserved verbatim, and the following `SELECT` is a separate statement. This is
    // the case the token-stream span most depends on — the other tests only cover a `;`
    // that is NOT a Semicolon token; here it genuinely is one, consumed below the top level.
    let trigger =
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN UPDATE u SET a = 1; DELETE FROM z; END";
    let sql = format!("{trigger}; SELECT 1");
    let ast = parse(&sql).expect("parse");
    assert_eq!(ast.statements.len(), 2);
    assert!(matches!(ast.statements[0], Statement::CreateTrigger(_)));
    assert_eq!(ast.statement_sources[0], trigger);
    assert_eq!(ast.statement_sources[1], "SELECT 1");
}

#[test]
fn create_trigger_on_target_carries_schema_qualifier() {
    // The ON-target is a `QualifiedName`: a bare name parses unqualified, and the
    // spec's recommended schema-qualified form (`ON main.tab`, lang_createtrigger.html
    // §7 "TEMP Triggers on Non-TEMP Tables") parses with the qualifier preserved.
    let bare = parse("CREATE TEMP TRIGGER trg AFTER INSERT ON tab BEGIN SELECT 1; END")
        .expect("bare ON-target parses");
    let Statement::CreateTrigger(ct) = &bare.statements[0] else {
        panic!("expected CREATE TRIGGER");
    };
    assert_eq!(ct.table.schema, None, "bare ON-target has no qualifier");
    assert_eq!(ct.table.name, "tab");

    let qualified = parse("CREATE TEMP TRIGGER trg AFTER INSERT ON main.tab BEGIN SELECT 1; END")
        .expect("qualified ON-target parses");
    let Statement::CreateTrigger(ct) = &qualified.statements[0] else {
        panic!("expected CREATE TRIGGER");
    };
    assert_eq!(ct.table.schema.as_deref(), Some("main"), "qualified ON-target keeps its schema");
    assert_eq!(ct.table.name, "tab");
}

#[test]
fn create_view_source_is_verbatim() {
    // CREATE VIEW is another sqlite_schema.sql target; its text must be verbatim too.
    let sql = "CREATE VIEW v AS SELECT a, b FROM t WHERE a > 0";
    let ast = parse(sql).expect("parse");
    assert!(matches!(ast.statements[0], Statement::CreateView(_)));
    assert_eq!(ast.statement_sources[0], sql);
}

#[test]
fn final_statement_trailing_trivia_without_terminator_excluded() {
    // The last statement has no terminating `;` and a trailing comment: the span still
    // ends at the last token, so the whitespace + comment after it are excluded.
    let ast = parse("SELECT 1  -- tail; not part of it\n").expect("parse");
    assert_eq!(ast.statements.len(), 1);
    assert_eq!(ast.statement_sources[0], "SELECT 1");
}

#[test]
fn statement_source_slices_multibyte_utf8_correctly() {
    // The span slices `src` by BYTE offset, so a non-ASCII literal must be preserved
    // whole. This would panic (mid-char boundary) or mis-slice if token offsets were
    // ever char indices rather than byte offsets — a guard on that tokenizer assumption.
    let sql = "INSERT INTO t VALUES ('café ; α;β')";
    let ast = parse(sql).expect("parse");
    assert_eq!(ast.statements.len(), 1);
    assert_eq!(ast.statement_sources[0], sql);
}
