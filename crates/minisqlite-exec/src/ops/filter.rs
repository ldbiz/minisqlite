//! `Filter` — keep only rows whose predicate is TRUE. Three-valued: a NULL or FALSE
//! predicate drops the row (only `Some(true)` survives). Passes surviving rows
//! through unchanged (input layout).

use minisqlite_expr::{eval, truth, EvalExpr};
use minisqlite_types::{Result, Row};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a filter over `input`, keeping rows where `predicate` evaluates TRUE.
pub(crate) fn filter<'a>(
    env: Env<'a>,
    predicate: &'a EvalExpr,
    input: Box<dyn RowCursor + 'a>,
) -> Box<dyn RowCursor + 'a> {
    Box::new(FilterCursor { env, predicate, input })
}

struct FilterCursor<'a> {
    env: Env<'a>,
    predicate: &'a EvalExpr,
    input: Box<dyn RowCursor + 'a>,
}

impl RowCursor for FilterCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        // Pull until a row passes or the input is exhausted. Bounded by the input.
        while let Some(row) = self.input.next_row(rt)? {
            let keep = {
                let mut ctx = EvalCtx { rt, env: self.env, outer: &row };
                truth(&eval(self.predicate, &row, &mut ctx)?) == Some(true)
            };
            if keep {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
}
