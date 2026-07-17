//! Leaf row sources with no table behind them: [`values`] (a literal `VALUES` / an
//! `INSERT` source) and [`single_row`] (a `SELECT` with no `FROM`).

use minisqlite_expr::{eval, EvalExpr};
use minisqlite_types::{Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::with_outer;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a `VALUES` source: each inner `Vec<EvalExpr>` is one row; every expression
/// is evaluated with the `outer` row as its registers. Emits `outer ++ local`.
pub(crate) fn values<'a>(
    env: Env<'a>,
    rows: &'a [Vec<EvalExpr>],
    outer: &'a [Value],
) -> Box<dyn RowCursor + 'a> {
    Box::new(ValuesCursor { env, rows, outer, idx: 0 })
}

struct ValuesCursor<'a> {
    env: Env<'a>,
    rows: &'a [Vec<EvalExpr>],
    outer: &'a [Value],
    idx: usize,
}

impl RowCursor for ValuesCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        let Some(exprs) = self.rows.get(self.idx) else {
            return Ok(None);
        };
        self.idx += 1;
        let mut local = Vec::with_capacity(exprs.len());
        let mut ctx = EvalCtx { rt, env: self.env, outer: self.outer };
        for e in exprs {
            local.push(eval(e, self.outer, &mut ctx)?);
        }
        Ok(Some(with_outer(self.outer, local)))
    }
}

/// Build a `SingleRow` source: exactly one row of zero columns (`outer ++ []`), then
/// end. The scalar projection above it supplies the actual output columns.
pub(crate) fn single_row<'a>(outer: &'a [Value]) -> Box<dyn RowCursor + 'a> {
    Box::new(SingleRowCursor { outer, done: false })
}

struct SingleRowCursor<'a> {
    outer: &'a [Value],
    done: bool,
}

impl RowCursor for SingleRowCursor<'_> {
    fn next_row(&mut self, _rt: &mut Runtime) -> Result<Option<Row>> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        Ok(Some(with_outer(self.outer, Vec::new())))
    }
}
