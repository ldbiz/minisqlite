//! `Project` — evaluate a list of expressions against each input row and emit them as
//! the new row. Output width = number of expressions.

use minisqlite_expr::{eval, EvalExpr};
use minisqlite_types::{Result, Row};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a projection of `exprs` over `input`.
pub(crate) fn project<'a>(
    env: Env<'a>,
    exprs: &'a [EvalExpr],
    input: Box<dyn RowCursor + 'a>,
) -> Box<dyn RowCursor + 'a> {
    Box::new(ProjectCursor { env, exprs, input })
}

struct ProjectCursor<'a> {
    env: Env<'a>,
    exprs: &'a [EvalExpr],
    input: Box<dyn RowCursor + 'a>,
}

impl RowCursor for ProjectCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        let Some(row) = self.input.next_row(rt)? else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(self.exprs.len());
        let mut ctx = EvalCtx { rt, env: self.env, outer: &row };
        for e in self.exprs {
            out.push(eval(e, &row, &mut ctx)?);
        }
        Ok(Some(out))
    }
}
