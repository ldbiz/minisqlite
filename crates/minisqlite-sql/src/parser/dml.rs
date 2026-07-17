//! DML parsing: INSERT, UPDATE, DELETE, with UPSERT and RETURNING.
//! Specs: `lang_insert.html`, `lang_update.html`, `lang_delete.html`,
//! `lang_upsert.html`, `lang_returning.html`.

use crate::ast_dml::{
    Delete, Insert, InsertSource, SetClause, Update, Upsert, UpsertAction, UpsertTarget,
};
use crate::ast_select::ResultColumn;
use crate::ast_stmt::{ConflictClause, With};
use crate::keyword::Keyword;
use crate::parser::Parser;
use crate::token::TokenKind;
use minisqlite_types::Result;

impl<'a> Parser<'a> {
    // --- INSERT ---------------------------------------------------------------

    pub(crate) fn parse_insert(&mut self, with: Option<With>) -> Result<Insert> {
        // `REPLACE INTO` is sugar for `INSERT OR REPLACE INTO`.
        let or_conflict = if self.eat_kw(Keyword::Replace) {
            Some(ConflictClause::Replace)
        } else {
            self.expect_kw(Keyword::Insert)?;
            if self.eat_kw(Keyword::Or) { Some(self.parse_conflict_algorithm()?) } else { None }
        };
        self.expect_kw(Keyword::Into)?;
        let table = self.parse_qualified_name()?;
        let alias = if self.eat_kw(Keyword::As) { Some(self.parse_ident()?) } else { None };
        let columns =
            if self.check(&TokenKind::LParen) { Some(self.parse_paren_name_list()?) } else { None };
        let source = self.parse_insert_source()?;
        let upsert = self.parse_upserts()?;
        let returning = self.parse_opt_returning()?;
        Ok(Insert { with, or_conflict, table, alias, columns, source, upsert, returning })
    }

    fn parse_insert_source(&mut self) -> Result<InsertSource> {
        if self.eat_kw2(Keyword::Default, Keyword::Values) {
            return Ok(InsertSource::DefaultValues);
        }
        if self.eat_kw(Keyword::Values) {
            let mut rows = vec![self.parse_paren_expr_row()?];
            while self.eat(&TokenKind::Comma) {
                rows.push(self.parse_paren_expr_row()?);
            }
            return Ok(InsertSource::Values(rows));
        }
        if self.check_kw(Keyword::Select) || self.check_kw(Keyword::With) {
            return Ok(InsertSource::Select(Box::new(self.parse_select()?)));
        }
        self.error("expected VALUES, SELECT, or DEFAULT VALUES after the INSERT target")
    }

    fn parse_upserts(&mut self) -> Result<Vec<Upsert>> {
        let mut upserts = Vec::new();
        while self.eat_kw2(Keyword::On, Keyword::Conflict) {
            upserts.push(self.parse_upsert_tail()?);
        }
        Ok(upserts)
    }

    /// The body after `ON CONFLICT` (already consumed).
    fn parse_upsert_tail(&mut self) -> Result<Upsert> {
        let target = if self.check(&TokenKind::LParen) {
            let columns = self.parse_indexed_column_list()?;
            let where_clause =
                if self.eat_kw(Keyword::Where) { Some(self.parse_expr()?) } else { None };
            Some(UpsertTarget { columns, where_clause })
        } else {
            None
        };
        self.expect_kw(Keyword::Do)?;
        let action = if self.eat_kw(Keyword::Nothing) {
            UpsertAction::Nothing
        } else if self.eat_kw(Keyword::Update) {
            self.expect_kw(Keyword::Set)?;
            let set = self.parse_set_clauses()?;
            let where_clause =
                if self.eat_kw(Keyword::Where) { Some(self.parse_expr()?) } else { None };
            UpsertAction::Update { set, where_clause }
        } else {
            return self.error("expected NOTHING or UPDATE after ON CONFLICT ... DO");
        };
        Ok(Upsert { target, action })
    }

    // --- UPDATE ---------------------------------------------------------------

    pub(crate) fn parse_update(&mut self, with: Option<With>) -> Result<Update> {
        self.expect_kw(Keyword::Update)?;
        let or_conflict =
            if self.eat_kw(Keyword::Or) { Some(self.parse_conflict_algorithm()?) } else { None };
        let table = self.parse_qualified_name()?;
        let alias = if self.eat_kw(Keyword::As) { Some(self.parse_ident()?) } else { None };
        let indexed = self.parse_opt_indexed()?;
        self.expect_kw(Keyword::Set)?;
        let set = self.parse_set_clauses()?;
        let from = if self.eat_kw(Keyword::From) { Some(self.parse_from()?) } else { None };
        let where_clause =
            if self.eat_kw(Keyword::Where) { Some(self.parse_expr()?) } else { None };
        let returning = self.parse_opt_returning()?;
        Ok(Update { with, or_conflict, table, alias, indexed, set, from, where_clause, returning })
    }

    fn parse_set_clauses(&mut self) -> Result<Vec<SetClause>> {
        let mut set = vec![self.parse_set_clause()?];
        while self.eat(&TokenKind::Comma) {
            set.push(self.parse_set_clause()?);
        }
        Ok(set)
    }

    fn parse_set_clause(&mut self) -> Result<SetClause> {
        if self.check(&TokenKind::LParen) {
            let names = self.parse_paren_name_list()?;
            self.expect(&TokenKind::Eq, "'=' in SET assignment")?;
            let value = self.parse_expr()?;
            Ok(SetClause::Columns { names, value })
        } else {
            let name = self.parse_ident()?;
            self.expect(&TokenKind::Eq, "'=' in SET assignment")?;
            let value = self.parse_expr()?;
            Ok(SetClause::Column { name, value })
        }
    }

    // --- DELETE ---------------------------------------------------------------

    pub(crate) fn parse_delete(&mut self, with: Option<With>) -> Result<Delete> {
        self.expect_kw(Keyword::Delete)?;
        self.expect_kw(Keyword::From)?;
        let table = self.parse_qualified_name()?;
        let alias = if self.eat_kw(Keyword::As) { Some(self.parse_ident()?) } else { None };
        let indexed = self.parse_opt_indexed()?;
        let where_clause =
            if self.eat_kw(Keyword::Where) { Some(self.parse_expr()?) } else { None };
        let returning = self.parse_opt_returning()?;
        Ok(Delete { with, table, alias, indexed, where_clause, returning })
    }

    // --- RETURNING ------------------------------------------------------------

    fn parse_opt_returning(&mut self) -> Result<Vec<ResultColumn>> {
        if self.eat_kw(Keyword::Returning) {
            self.parse_result_columns()
        } else {
            Ok(Vec::new())
        }
    }
}
