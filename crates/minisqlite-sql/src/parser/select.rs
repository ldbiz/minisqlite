//! SELECT / VALUES parsing: the query body, compound set operators, result
//! columns, the FROM join tree, `WITH`, `ORDER BY`, and `LIMIT`.
//! Spec: `spec/sqlite-doc/lang_select.html`, `lang_with.html`.

use crate::ast_expr::{Expr, WindowSpec};
use crate::ast_select::{
    CompoundOp, Distinct, FromClause, IndexedClause, JoinConstraint, JoinKind, JoinOperator,
    JoinTree, Limit, NullsOrder, OrderingTerm, ResultColumn, Select, SelectBody, SelectCore,
    SortOrder, TableOrSubquery,
};
use crate::ast_stmt::{Cte, With};
use crate::keyword::Keyword;
use crate::parser::{Parser, MAX_COMPOUND_SELECT_TERMS, MAX_JOIN_TABLES};
use crate::token::TokenKind;
use minisqlite_types::Result;

impl<'a> Parser<'a> {
    // --- WITH -----------------------------------------------------------------

    /// `WITH [RECURSIVE] cte, cte, ...`.
    pub(crate) fn parse_with(&mut self) -> Result<With> {
        self.expect_kw(Keyword::With)?;
        let recursive = self.eat_kw(Keyword::Recursive);
        let mut ctes = vec![self.parse_cte()?];
        while self.eat(&TokenKind::Comma) {
            ctes.push(self.parse_cte()?);
        }
        Ok(With { recursive, ctes })
    }

    fn parse_cte(&mut self) -> Result<Cte> {
        let name = self.parse_ident()?;
        let columns =
            if self.check(&TokenKind::LParen) { Some(self.parse_paren_name_list()?) } else { None };
        self.expect_kw(Keyword::As)?;
        let materialized = if self.eat_kw(Keyword::Materialized) {
            Some(true)
        } else if self.eat_kw2(Keyword::Not, Keyword::Materialized) {
            Some(false)
        } else {
            None
        };
        self.expect(&TokenKind::LParen, "'(' before common-table-expression query")?;
        let select = self.parse_select()?;
        self.expect(&TokenKind::RParen, "')' after common-table-expression query")?;
        Ok(Cte { name, columns, materialized, select: Box::new(select) })
    }

    // --- SELECT ---------------------------------------------------------------

    /// A full SELECT statement, including its own optional leading `WITH`. Every
    /// nested query (subquery, CTE body, `FROM` subquery, view/trigger body)
    /// routes through here, so it guards recursion depth (see `MAX_PARSE_DEPTH`).
    pub(crate) fn parse_select(&mut self) -> Result<Select> {
        self.enter()?;
        let r = self.parse_select_inner();
        self.leave();
        r
    }

    fn parse_select_inner(&mut self) -> Result<Select> {
        let with = if self.check_kw(Keyword::With) { Some(self.parse_with()?) } else { None };
        self.parse_select_after_with(with)
    }

    /// A SELECT whose `WITH` (if any) has already been parsed.
    pub(crate) fn parse_select_after_with(&mut self, with: Option<With>) -> Result<Select> {
        let body = self.parse_select_body()?;
        let order_by = if self.eat_kw2(Keyword::Order, Keyword::By) {
            self.parse_ordering_terms()?
        } else {
            Vec::new()
        };
        let limit = if self.eat_kw(Keyword::Limit) { Some(self.parse_limit()?) } else { None };
        Ok(Select { with, body, order_by, limit })
    }

    /// A query body: one core, then any compound set operators, left-associative.
    /// The term count is bounded (see [`MAX_COMPOUND_SELECT_TERMS`]) so an
    /// oversized chain errors like SQLite instead of building an unbounded tree.
    fn parse_select_body(&mut self) -> Result<SelectBody> {
        let mut left = SelectBody::Select(self.parse_select_core()?);
        let mut terms: u32 = 1;
        loop {
            let op = if self.eat_kw(Keyword::Union) {
                if self.eat_kw(Keyword::All) {
                    CompoundOp::UnionAll
                } else {
                    CompoundOp::Union
                }
            } else if self.eat_kw(Keyword::Intersect) {
                CompoundOp::Intersect
            } else if self.eat_kw(Keyword::Except) {
                CompoundOp::Except
            } else {
                break;
            };
            terms += 1;
            self.enforce_max(terms, MAX_COMPOUND_SELECT_TERMS, "too many terms in compound SELECT")?;
            let right = self.parse_select_core()?;
            left = SelectBody::Compound { op, left: Box::new(left), right };
        }
        Ok(left)
    }

    fn parse_select_core(&mut self) -> Result<SelectCore> {
        if self.eat_kw(Keyword::Values) {
            let mut rows = vec![self.parse_paren_expr_row()?];
            while self.eat(&TokenKind::Comma) {
                rows.push(self.parse_paren_expr_row()?);
            }
            return Ok(SelectCore::Values(rows));
        }
        self.expect_kw(Keyword::Select)?;
        let distinct = if self.eat_kw(Keyword::Distinct) {
            Distinct::Distinct
        } else {
            self.eat_kw(Keyword::All);
            Distinct::All
        };
        let columns = self.parse_result_columns()?;
        let from = if self.eat_kw(Keyword::From) { Some(self.parse_from()?) } else { None };
        let where_clause =
            if self.eat_kw(Keyword::Where) { Some(self.parse_expr()?) } else { None };
        let group_by = if self.eat_kw2(Keyword::Group, Keyword::By) {
            let mut v = vec![self.parse_expr()?];
            while self.eat(&TokenKind::Comma) {
                v.push(self.parse_expr()?);
            }
            v
        } else {
            Vec::new()
        };
        let having = if self.eat_kw(Keyword::Having) { Some(self.parse_expr()?) } else { None };
        let windows =
            if self.eat_kw(Keyword::Window) { self.parse_window_defs()? } else { Vec::new() };
        Ok(SelectCore::Query { distinct, columns, from, where_clause, group_by, having, windows })
    }

    /// `( expr, expr, ... )` — one VALUES row.
    pub(crate) fn parse_paren_expr_row(&mut self) -> Result<Vec<Expr>> {
        self.expect(&TokenKind::LParen, "'(' before a VALUES row")?;
        let mut row = vec![self.parse_expr()?];
        while self.eat(&TokenKind::Comma) {
            row.push(self.parse_expr()?);
        }
        self.expect(&TokenKind::RParen, "')' after a VALUES row")?;
        Ok(row)
    }

    fn parse_window_defs(&mut self) -> Result<Vec<(String, WindowSpec)>> {
        let mut defs = Vec::new();
        loop {
            let name = self.parse_ident()?;
            self.expect_kw(Keyword::As)?;
            let spec = self.parse_window_spec()?;
            defs.push((name, spec));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        Ok(defs)
    }

    // --- result columns -------------------------------------------------------

    pub(crate) fn parse_result_columns(&mut self) -> Result<Vec<ResultColumn>> {
        let mut cols = vec![self.parse_result_column()?];
        while self.eat(&TokenKind::Comma) {
            cols.push(self.parse_result_column()?);
        }
        Ok(cols)
    }

    fn parse_result_column(&mut self) -> Result<ResultColumn> {
        if self.eat(&TokenKind::Star) {
            return Ok(ResultColumn::Star);
        }
        // `table.*`
        if self.at_ident()
            && *self.kind_at(1) == TokenKind::Dot
            && *self.kind_at(2) == TokenKind::Star
        {
            let table = self.parse_ident()?;
            self.bump(); // '.'
            self.bump(); // '*'
            return Ok(ResultColumn::TableStar(table));
        }
        let expr = self.parse_expr()?;
        let alias = self.parse_opt_alias(false)?;
        Ok(ResultColumn::Expr { expr, alias })
    }

    /// An optional `[AS] alias`. `exclude_indexed` avoids swallowing an `INDEXED`
    /// keyword (which begins a table's index hint) as an implicit table alias.
    ///
    /// The `AS`-prefixed form takes any name (`nm`: identifier, string, or a
    /// non-reserved keyword incl. join words — `t AS left`). The bare (no-`AS`) form is
    /// narrower, matching SQLite's `as ::= ids` rule plus its tokenizer look-ahead: it
    /// refuses a join keyword (which always introduces a join here) and a
    /// `WINDOW <name> AS` window-definition clause, so `FROM a LEFT JOIN b` and
    /// `FROM t WINDOW w AS (…)` are never mis-read as `a`/`t` aliased.
    pub(crate) fn parse_opt_alias(&mut self, exclude_indexed: bool) -> Result<Option<String>> {
        if self.eat_kw(Keyword::As) {
            if let TokenKind::Str(s) = self.kind().clone() {
                self.bump();
                return Ok(Some(s));
            }
            return Ok(Some(self.parse_ident()?));
        }
        if self.at_bare_alias(exclude_indexed) {
            return Ok(Some(self.parse_ident()?));
        }
        Ok(None)
    }

    /// Whether the current token may be consumed as a bare (no-`AS`) alias. Requires an
    /// identifier-position token, then rules out the context-sensitive keywords that in
    /// this position belong to the enclosing grammar, not to an alias (see
    /// [`parse_opt_alias`]).
    fn at_bare_alias(&self, exclude_indexed: bool) -> bool {
        if !self.at_ident() {
            return false;
        }
        match self.kw() {
            // A join word here starts a join (`FROM a LEFT JOIN b`), never an alias.
            Some(kw) if kw.is_join_keyword() => false,
            // INDEXED begins a table's `INDEXED BY` hint (table-source context only).
            Some(Keyword::Indexed) if exclude_indexed => false,
            // `WINDOW <name> AS` is a window-definition clause, not an alias — the same
            // `<id> AS` look-ahead SQLite's `analyzeWindowKeyword` uses. A lone `WINDOW`
            // not so followed remains a usable alias.
            Some(Keyword::Window)
                if self.at_ident_offset(1) && self.kw_at(2) == Some(Keyword::As) =>
            {
                false
            }
            _ => true,
        }
    }

    /// Whether the token `off` ahead is in identifier position (an identifier or a
    /// non-reserved keyword). A small look-ahead helper for [`at_bare_alias`].
    fn at_ident_offset(&self, off: usize) -> bool {
        match self.kind_at(off) {
            TokenKind::Ident | TokenKind::QuotedIdent { .. } => true,
            TokenKind::Keyword(kw) => kw.can_be_identifier(),
            _ => false,
        }
    }

    // --- FROM join tree -------------------------------------------------------

    /// The FROM clause: one table/subquery, then any `,`/`JOIN` operands folded
    /// left-associatively. The table count is bounded (see [`MAX_JOIN_TABLES`]) so
    /// an oversized join errors like SQLite instead of building an unbounded tree.
    pub(crate) fn parse_from(&mut self) -> Result<FromClause> {
        let mut left = JoinTree::Table(self.parse_table_or_subquery()?);
        let mut tables: u32 = 1;
        loop {
            if self.eat(&TokenKind::Comma) {
                tables += 1;
                self.enforce_max(tables, MAX_JOIN_TABLES, "at most 64 tables in a join")?;
                let right = self.parse_table_or_subquery()?;
                left = JoinTree::Join {
                    left: Box::new(left),
                    op: JoinOperator { natural: false, kind: JoinKind::Comma },
                    right,
                    constraint: None,
                };
                continue;
            }
            if let Some(op) = self.try_parse_join_operator()? {
                tables += 1;
                self.enforce_max(tables, MAX_JOIN_TABLES, "at most 64 tables in a join")?;
                let right = self.parse_table_or_subquery()?;
                let constraint = self.parse_opt_join_constraint()?;
                left = JoinTree::Join { left: Box::new(left), op, right, constraint };
                continue;
            }
            break;
        }
        Ok(left)
    }

    /// Parse an explicit join operator if one starts here. Returns `None` (consuming
    /// nothing) when the next tokens are not a join operator.
    fn try_parse_join_operator(&mut self) -> Result<Option<JoinOperator>> {
        let natural = self.check_kw(Keyword::Natural);
        let head = self.kw_at(if natural { 1 } else { 0 });
        let kind = match head {
            Some(Keyword::Join) => JoinKind::Inner,
            Some(Keyword::Inner) => JoinKind::Inner,
            Some(Keyword::Left) => JoinKind::Left,
            Some(Keyword::Right) => JoinKind::Right,
            Some(Keyword::Full) => JoinKind::Full,
            Some(Keyword::Cross) if !natural => JoinKind::Cross,
            _ => return Ok(None),
        };
        if natural {
            self.bump();
        }
        // Consume the operator words, then the mandatory JOIN keyword.
        match kind {
            JoinKind::Inner if self.check_kw(Keyword::Join) => {} // bare JOIN
            JoinKind::Inner => {
                self.expect_kw(Keyword::Inner)?;
            }
            JoinKind::Left | JoinKind::Right | JoinKind::Full => {
                self.bump(); // LEFT / RIGHT / FULL
                self.eat_kw(Keyword::Outer);
            }
            JoinKind::Cross => {
                self.bump(); // CROSS
            }
            JoinKind::Comma => unreachable!("comma joins are handled separately"),
        }
        self.expect_kw(Keyword::Join)?;
        Ok(Some(JoinOperator { natural, kind }))
    }

    fn parse_opt_join_constraint(&mut self) -> Result<Option<JoinConstraint>> {
        if self.eat_kw(Keyword::On) {
            Ok(Some(JoinConstraint::On(self.parse_expr()?)))
        } else if self.eat_kw(Keyword::Using) {
            Ok(Some(JoinConstraint::Using(self.parse_paren_name_list()?)))
        } else {
            Ok(None)
        }
    }

    /// One FROM source. A parenthesized join clause recurses back into
    /// `parse_from` here, so this is the join-nesting recursion head and guards
    /// depth (see `MAX_PARSE_DEPTH`).
    fn parse_table_or_subquery(&mut self) -> Result<TableOrSubquery> {
        self.enter()?;
        let r = self.parse_table_or_subquery_inner();
        self.leave();
        r
    }

    fn parse_table_or_subquery_inner(&mut self) -> Result<TableOrSubquery> {
        if self.eat(&TokenKind::LParen) {
            if self.at_select_start() {
                let sel = self.parse_select()?;
                self.expect(&TokenKind::RParen, "')' after subquery")?;
                let alias = self.parse_opt_alias(false)?;
                return Ok(TableOrSubquery::Subquery { select: Box::new(sel), alias });
            }
            let inner = self.parse_from()?;
            self.expect(&TokenKind::RParen, "')' after parenthesized join")?;
            return Ok(TableOrSubquery::Join(Box::new(inner)));
        }
        let name = self.parse_qualified_name()?;
        if self.eat(&TokenKind::LParen) {
            let mut args = Vec::new();
            if !self.check(&TokenKind::RParen) {
                args.push(self.parse_expr()?);
                while self.eat(&TokenKind::Comma) {
                    args.push(self.parse_expr()?);
                }
            }
            self.expect(&TokenKind::RParen, "')' after table-function arguments")?;
            let alias = self.parse_opt_alias(false)?;
            return Ok(TableOrSubquery::TableFunction { name, args, alias });
        }
        let alias = self.parse_opt_alias(true)?;
        let indexed = self.parse_opt_indexed()?;
        Ok(TableOrSubquery::Table { name, alias, indexed })
    }

    pub(crate) fn parse_opt_indexed(&mut self) -> Result<Option<IndexedClause>> {
        if self.eat_kw(Keyword::Indexed) {
            self.expect_kw(Keyword::By)?;
            Ok(Some(IndexedClause::IndexedBy(self.parse_ident()?)))
        } else if self.eat_kw2(Keyword::Not, Keyword::Indexed) {
            Ok(Some(IndexedClause::NotIndexed))
        } else {
            Ok(None)
        }
    }

    // --- ORDER BY / LIMIT -----------------------------------------------------

    pub(crate) fn parse_ordering_terms(&mut self) -> Result<Vec<OrderingTerm>> {
        let mut terms = vec![self.parse_ordering_term()?];
        while self.eat(&TokenKind::Comma) {
            terms.push(self.parse_ordering_term()?);
        }
        Ok(terms)
    }

    fn parse_ordering_term(&mut self) -> Result<OrderingTerm> {
        // `COLLATE` binds as an expression operator, so an explicit collation here
        // is usually already folded into `expr`; the field stays for completeness.
        let expr = self.parse_expr()?;
        let collation =
            if self.eat_kw(Keyword::Collate) { Some(self.parse_ident()?) } else { None };
        let order = if self.eat_kw(Keyword::Asc) {
            Some(SortOrder::Asc)
        } else if self.eat_kw(Keyword::Desc) {
            Some(SortOrder::Desc)
        } else {
            None
        };
        let nulls = if self.eat_kw2(Keyword::Nulls, Keyword::First) {
            Some(NullsOrder::First)
        } else if self.eat_kw2(Keyword::Nulls, Keyword::Last) {
            Some(NullsOrder::Last)
        } else {
            None
        };
        Ok(OrderingTerm { expr, collation, order, nulls })
    }

    fn parse_limit(&mut self) -> Result<Limit> {
        let first = self.parse_expr()?;
        if self.eat_kw(Keyword::Offset) {
            let offset = self.parse_expr()?;
            Ok(Limit { limit: first, offset: Some(offset) })
        } else if self.eat(&TokenKind::Comma) {
            // `LIMIT offset, limit` — SQLite's comma form puts the offset first.
            let limit = self.parse_expr()?;
            Ok(Limit { limit, offset: Some(first) })
        } else {
            Ok(Limit { limit: first, offset: None })
        }
    }

    /// `( name, name, ... )` — a parenthesized identifier list.
    pub(crate) fn parse_paren_name_list(&mut self) -> Result<Vec<String>> {
        self.expect(&TokenKind::LParen, "'('")?;
        let mut names = vec![self.parse_ident()?];
        while self.eat(&TokenKind::Comma) {
            names.push(self.parse_ident()?);
        }
        self.expect(&TokenKind::RParen, "')'")?;
        Ok(names)
    }
}
