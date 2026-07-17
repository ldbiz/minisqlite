//! The expression evaluator: one pass over an [`EvalExpr`] tree per row.
//!
//! Semantics follow lang_expr.html §1-§8 and datatype3.html §4/§8: three-valued
//! logic (NULL = unknown), SQLite's implicit numeric coercions at the operators,
//! affinity applied to comparison operands, and the short-circuiting logical
//! connectives. The rules are transcribed here and pinned by the tests at the
//! bottom; where a subtle value is expected, the test states it exactly.
//!
//! The evaluator never panics on any input: an out-of-range register or parameter
//! (which the binder should never emit) is a returned `Err`, not a crash. Recursion
//! mirrors the bound expression depth, which the SQL parser bounds upstream.

use std::cmp::Ordering;

use minisqlite_types::{apply_affinity, cast_to, compare_for_eq, Affinity, Error, Result, Value};

use crate::coerce::{as_num, to_integer, truth, Num};
use crate::context::{EvalContext, FnContext};
use crate::datetime::format_now;
use crate::ir::{ArithOp, BitOp, CmpOp, CompareMeta, EvalExpr, LikeKind, RaiseKind, UnaryOp};
use crate::pattern::{glob_matches, like_matches};

/// Evaluate a bound expression against the current row `regs`, reaching out through
/// `ctx` for parameters, subqueries, the clock, and RNG. Returns the resulting
/// [`Value`] (which may be NULL) or an error for a genuinely invalid expression.
///
/// Public entry point: it discards the ephemeral value-subtype the JSON functions
/// ride on (see [`eval_with_subtype`]) — a value handed back to an operator and
/// stored to a row carries no subtype (json1.html §3.4).
pub fn eval(expr: &EvalExpr, regs: &[Value], ctx: &mut dyn EvalContext) -> Result<Value> {
    eval_with_subtype(expr, regs, ctx).map(|(v, _)| v)
}

/// Evaluate `expr`, additionally returning the ephemeral value-subtype of its result
/// (`0` = none). The subtype is non-zero ONLY for a function call whose function set
/// one via [`FnContext::set_result_subtype`] — the JSON functions do, so a nested JSON
/// result is embedded rather than re-quoted (json1.html §3.4); every other expression
/// yields subtype `0`.
///
/// The threading lives in this one place so it is captured correctly: a function's
/// arguments are each evaluated through THIS function, and each argument's subtype is
/// captured IMMEDIATELY into `arg_subtypes` (via the tuple return) so evaluating a
/// later argument cannot clobber an earlier one's subtype in the single-slot context
/// channel. The evaluator then publishes the collected arg subtypes, calls the
/// function, and reads back the function's own result subtype.
///
/// Public so the executor's AGGREGATE / WINDOW drivers — which evaluate aggregate
/// arguments themselves rather than through the [`EvalExpr::Func`] arm above — can
/// capture each argument's subtype the same way and publish it to the accumulator's
/// `step` (an aggregate embeds a subtyped `value` operand exactly as a scalar does).
pub fn eval_with_subtype(
    expr: &EvalExpr,
    regs: &[Value],
    ctx: &mut dyn EvalContext,
) -> Result<(Value, u8)> {
    let EvalExpr::Func { func, args } = expr else {
        // Every non-function node is subtype-less. A function nested inside it still
        // threads its OWN arguments' subtypes (each sub-expression is evaluated via
        // `eval`, which routes back here), but the enclosing non-function node does
        // not propagate a subtype outward.
        return eval_plain(expr, regs, ctx).map(|v| (v, 0));
    };
    // Evaluate each argument, capturing its subtype immediately. `arg_subtypes` is
    // filled LAZILY: it stays empty (no allocation) unless some argument actually
    // carries a subtype — the overwhelmingly common case for non-JSON calls, keeping
    // ordinary per-row function evaluation allocation-free on this path.
    let mut argv = Vec::with_capacity(args.len());
    let mut arg_subtypes: Vec<u8> = Vec::new();
    let mut any_subtype = false;
    for (i, a) in args.iter().enumerate() {
        let (v, st) = eval_with_subtype(a, regs, ctx)?;
        if st != 0 && !any_subtype {
            // First subtype seen: back-fill zeros for the arguments already pushed so
            // subtype index `i` lines up with argument `i`, then start recording.
            any_subtype = true;
            arg_subtypes = vec![0u8; i];
        }
        if any_subtype {
            arg_subtypes.push(st);
        }
        argv.push(v);
    }
    // Publish the arg subtypes (an EMPTY slice clears any prior call's, so every
    // `arg_subtype(i)` reads 0) and clear any stale result subtype, then call. `ctx`
    // upcasts from `&mut dyn EvalContext` to the smaller `&mut dyn FnContext`.
    let fnctx: &mut dyn FnContext = ctx;
    fnctx.set_result_subtype(0);
    fnctx.set_arg_subtypes(&arg_subtypes);
    // Each function decides its own NULL behavior — do not pre-check NULLs.
    let out = func.call(&argv, fnctx)?;
    let st = fnctx.take_result_subtype();
    Ok((out, st))
}

/// Evaluate a bound expression to a [`Value`], ignoring value-subtypes (the subtype
/// wiring lives in [`eval_with_subtype`]). Reached for every non-function node; the
/// `Func` arm is never taken here (that node is intercepted by `eval_with_subtype`)
/// but delegates back so function evaluation stays a single code path.
fn eval_plain(expr: &EvalExpr, regs: &[Value], ctx: &mut dyn EvalContext) -> Result<Value> {
    match expr {
        EvalExpr::Literal(v) => Ok(v.clone()),

        EvalExpr::Column(i) => regs
            .get(*i)
            .cloned()
            .ok_or_else(|| Error::sql("column register index out of range")),

        EvalExpr::Param(i) => ctx.param(*i),

        EvalExpr::Now(kind) => Ok(Value::Text(format_now(*kind, ctx.now_unix_millis()))),

        EvalExpr::Unary { op, operand } => {
            let v = eval(operand, regs, ctx)?;
            Ok(unary(*op, v))
        }

        EvalExpr::Arith { op, left, right } => {
            let l = eval(left, regs, ctx)?;
            let r = eval(right, regs, ctx)?;
            Ok(arith_op(*op, l, r))
        }

        EvalExpr::Concat { left, right } => {
            let l = eval(left, regs, ctx)?;
            let r = eval(right, regs, ctx)?;
            Ok(concat(l, r))
        }

        EvalExpr::Bitwise { op, left, right } => {
            let l = eval(left, regs, ctx)?;
            let r = eval(right, regs, ctx)?;
            Ok(bit_op(*op, l, r))
        }

        EvalExpr::Compare { op, null_safe, left, right, meta } => {
            let lv = eval(left, regs, ctx)?;
            let rv = eval(right, regs, ctx)?;
            Ok(compare_op(*op, *null_safe, lv, rv, meta))
        }

        EvalExpr::And(a, b) => {
            // Short-circuit: a false operand makes AND false without evaluating the
            // other; otherwise the result is 1 only if both are true, else NULL.
            let ta = truth(&eval(a, regs, ctx)?);
            if ta == Some(false) {
                return Ok(Value::Integer(0));
            }
            let tb = truth(&eval(b, regs, ctx)?);
            if tb == Some(false) {
                return Ok(Value::Integer(0));
            }
            Ok(bool_to_value(match (ta, tb) {
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }))
        }

        EvalExpr::Or(a, b) => {
            let ta = truth(&eval(a, regs, ctx)?);
            if ta == Some(true) {
                return Ok(Value::Integer(1));
            }
            let tb = truth(&eval(b, regs, ctx)?);
            if tb == Some(true) {
                return Ok(Value::Integer(1));
            }
            Ok(bool_to_value(match (ta, tb) {
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }))
        }

        EvalExpr::IsNull(x) => {
            let is_null = eval(x, regs, ctx)?.is_null();
            Ok(Value::Integer(is_null as i64))
        }

        EvalExpr::NotNull(x) => {
            let is_null = eval(x, regs, ctx)?.is_null();
            Ok(Value::Integer((!is_null) as i64))
        }

        EvalExpr::Between { negated, subject, low, high, low_meta, high_meta } => {
            let v = eval(subject, regs, ctx)?; // subject evaluated once
            let lo = eval(low, regs, ctx)?;
            let hi = eval(high, regs, ctx)?;
            let ge = compare_meta(&v, &lo, low_meta)
                .map(|o| matches!(o, Ordering::Greater | Ordering::Equal));
            let le = compare_meta(&v, &hi, high_meta)
                .map(|o| matches!(o, Ordering::Less | Ordering::Equal));
            let res = and3(ge, le);
            Ok(finish3(*negated, res))
        }

        EvalExpr::InList { negated, subject, items, meta } => {
            let v = eval(subject, regs, ctx)?;
            // `x IN ()` is false and `x NOT IN ()` is true — even for a NULL `x`.
            if items.is_empty() {
                return Ok(Value::Integer(if *negated { 1 } else { 0 }));
            }
            // Apply the probe's affinity once, then reuse it for every comparison.
            let probe = apply_opt(v, meta.apply_left);
            let mut found = false;
            let mut saw_null = false;
            for item in items {
                let iv = apply_opt(eval(item, regs, ctx)?, meta.apply_right);
                match compare_for_eq(&probe, &iv, meta.collation) {
                    Some(Ordering::Equal) => {
                        found = true;
                        break;
                    }
                    Some(_) => {}
                    None => saw_null = true,
                }
            }
            let res = if found {
                Some(true)
            } else if saw_null {
                None
            } else {
                Some(false)
            };
            Ok(finish3(*negated, res))
        }

        EvalExpr::InSubquery { negated, subject, id, meta } => {
            let v = eval(subject, regs, ctx)?;
            let r = ctx.eval_in_subquery(*id, &v, meta, regs)?;
            Ok(finish3(*negated, r))
        }

        EvalExpr::InSubqueryRow { negated, subjects, id, metas } => {
            // Evaluate each subject element against the row into the probe tuple, then
            // hand the whole tuple to the row-value IN callback. The negate-then-lower
            // (NULL stays NULL) rule is identical to the scalar `InSubquery` arm, via
            // `finish3`; the tuple three-valued membership itself lives in the context.
            let mut probe = Vec::with_capacity(subjects.len());
            for s in subjects {
                probe.push(eval(s, regs, ctx)?);
            }
            let r = ctx.eval_in_subquery_row(*id, &probe, metas, regs)?;
            Ok(finish3(*negated, r))
        }

        EvalExpr::Exists { negated, id } => {
            let b = ctx.eval_exists(*id, regs)?;
            Ok(Value::Integer((if *negated { !b } else { b }) as i64))
        }

        EvalExpr::ScalarSubquery(id) => ctx.eval_scalar_subquery(*id, regs),

        EvalExpr::ScalarSubqueryColumn { id, col } => {
            ctx.eval_scalar_subquery_column(*id, *col, regs)
        }

        EvalExpr::Coalesce(items) => {
            for item in items {
                let v = eval(item, regs, ctx)?;
                if !v.is_null() {
                    return Ok(v);
                }
            }
            Ok(Value::Null)
        }

        EvalExpr::NullIf { left, right, meta } => {
            let lv = eval(left, regs, ctx)?;
            let rv = eval(right, regs, ctx)?;
            if compare_meta(&lv, &rv, meta) == Some(Ordering::Equal) {
                Ok(Value::Null)
            } else {
                Ok(lv) // the ORIGINAL left value, not the affinity-applied copy
            }
        }

        EvalExpr::Case { operand, whens, else_expr } => {
            let ov = match operand {
                Some(o) => Some(eval(o, regs, ctx)?),
                None => None,
            };
            for w in whens {
                let matched = match &w.cmp {
                    Some(m) => {
                        let operand = ov.as_ref().ok_or_else(|| {
                            Error::sql("simple CASE arm without a CASE operand")
                        })?;
                        let wv = eval(&w.when, regs, ctx)?;
                        compare_meta(operand, &wv, m) == Some(Ordering::Equal)
                    }
                    None => truth(&eval(&w.when, regs, ctx)?) == Some(true),
                };
                if matched {
                    return eval(&w.then, regs, ctx);
                }
            }
            match else_expr {
                Some(e) => eval(e, regs, ctx),
                None => Ok(Value::Null),
            }
        }

        EvalExpr::Cast { affinity, operand } => Ok(cast_to(eval(operand, regs, ctx)?, *affinity)),

        // COLLATE is a pass-through at eval time: the binder has already folded the
        // collation into the enclosing comparison/sort metadata.
        EvalExpr::Collate { operand, .. } => eval(operand, regs, ctx),

        EvalExpr::Like { negated, kind, subject, pattern, escape } => {
            // GLOB has no ESCAPE clause; the binder must never attach one. If it did,
            // the escape would be evaluated (and could error/NULL) below and then
            // ignored by the GLOB matcher — a silent binder bug this catches in tests.
            debug_assert!(
                matches!(kind, LikeKind::Like) || escape.is_none(),
                "GLOB node carries an ESCAPE expression; only LIKE takes ESCAPE"
            );
            let sv = eval(subject, regs, ctx)?;
            let pv = eval(pattern, regs, ctx)?;
            if sv.is_null() || pv.is_null() {
                return Ok(Value::Null);
            }
            let esc = match escape {
                Some(e) => {
                    let ev = eval(e, regs, ctx)?;
                    if ev.is_null() {
                        return Ok(Value::Null);
                    }
                    let et = as_text(ev);
                    let mut chars = et.chars();
                    match (chars.next(), chars.next()) {
                        (Some(c), None) => Some(c),
                        _ => {
                            return Err(Error::sql("ESCAPE expression must be a single character"))
                        }
                    }
                }
                None => None,
            };
            let s = as_text(sv);
            let p = as_text(pv);
            let matched = match kind {
                LikeKind::Like => like_matches(&s, &p, esc),
                // GLOB has no ESCAPE clause in SQL; `esc` is always None here.
                LikeKind::Glob => glob_matches(&s, &p),
            };
            Ok(Value::Integer((matched ^ *negated) as i64))
        }

        // Intercepted by `eval_with_subtype` before it reaches here (so nested JSON
        // subtypes are threaded); this arm only satisfies exhaustiveness and routes
        // back to the single function-calling path, discarding the subtype.
        EvalExpr::Func { .. } => eval_with_subtype(expr, regs, ctx).map(|(v, _)| v),

        // RAISE(...) inside a trigger body (lang_createtrigger.html §6). ABORT/FAIL/
        // ROLLBACK terminate the current statement with an SQLITE_CONSTRAINT error
        // carrying the RAISE message verbatim (SQLite returns the raw message text; the
        // error KIND is what matters, not the wording). The error propagates
        // out of the firing action, and the engine's implicit transaction rolls the
        // statement back — the atomicity the spec requires.
        //
        // RAISE(IGNORE) is a distinct CONTROL SIGNAL ("abandon the current row's operation
        // and the rest of this trigger program, but continue the statement — no error, no
        // rollback"). The pinned 4-variant `Error` cannot carry a non-error skip, so the
        // signal rides a runtime flag instead: `signal_raise_ignore()` records the request
        // on the executor's context, and we return a SENTINEL `Err` purely to unwind out of
        // the trigger body. The enclosing `fire_triggers` consumes the flag and turns it
        // into a row-skip; only the executor context sets the flag, so in any other context
        // (a bare `SELECT RAISE(IGNORE)`, a test mock) the default no-op leaves this an
        // ordinary error — correct where there is no row to skip.
        EvalExpr::Raise { kind, message } => match kind {
            RaiseKind::Abort | RaiseKind::Fail | RaiseKind::Rollback => {
                Err(Error::Constraint(message.clone().unwrap_or_default()))
            }
            RaiseKind::Ignore => {
                ctx.signal_raise_ignore();
                Err(Error::sql("RAISE(IGNORE)"))
            }
        },
    }
}

/// Map a three-valued boolean to a SQL value: true -> 1, false -> 0, unknown -> NULL.
fn bool_to_value(b: Option<bool>) -> Value {
    match b {
        Some(true) => Value::Integer(1),
        Some(false) => Value::Integer(0),
        None => Value::Null,
    }
}

/// Three-valued AND over `Option<bool>`: a `false` on either side wins; both `true`
/// is `true`; anything else (a NULL meeting a non-false) is unknown.
fn and3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// Three-valued NOT: flips a known truth, leaves unknown unknown.
fn not3(a: Option<bool>) -> Option<bool> {
    a.map(|b| !b)
}

/// Finish a `[NOT] <predicate>` node: apply SQL `NOT` to a three-valued predicate
/// result (NULL stays NULL) when `negated`, then lower it to a SQL value. Shared by
/// BETWEEN / IN-list / IN-subquery so the negate-then-lower rule has a single home;
/// a change to negation/NULL semantics touches one place, not three.
fn finish3(negated: bool, res: Option<bool>) -> Value {
    bool_to_value(if negated { not3(res) } else { res })
}

/// Extract a text value's `String` after casting to TEXT affinity. Only ever called
/// on non-NULL values (callers handle NULL), so the non-Text arm is unreachable:
/// `cast_to(v, Text)` yields `Text` for every non-NULL `v` (INTEGER/REAL/TEXT/BLOB
/// all convert), and returns NULL only for NULL. The empty-string fallback keeps the
/// function total and panic-free; the `debug_assert` turns a caller that forgot to
/// NULL-check — the one way to reach that fallback — into a loud test failure instead
/// of a silent `""`.
fn as_text(v: Value) -> String {
    debug_assert!(!v.is_null(), "as_text on NULL: callers must NULL-check first");
    match cast_to(v, Affinity::Text) {
        Value::Text(s) => s,
        _ => String::new(),
    }
}

/// Apply an optional affinity to a value by move (no clone). `None` leaves it
/// untouched.
fn apply_opt(v: Value, a: Option<Affinity>) -> Value {
    match a {
        Some(af) => apply_affinity(v, af),
        None => v,
    }
}

/// Apply each operand's affinity (cloning only when an affinity is actually
/// applied) and compare with three-valued equality semantics. Returns `None` if
/// either post-affinity operand is NULL. Used by the nodes that compare borrowed
/// values (BETWEEN, NULLIF, simple CASE); the hot [`compare_op`] path takes owned
/// values and applies affinity by move instead.
fn compare_meta(a: &Value, b: &Value, meta: &CompareMeta) -> Option<Ordering> {
    match (meta.apply_left, meta.apply_right) {
        (None, None) => compare_for_eq(a, b, meta.collation),
        (la, ra) => {
            let av = apply_opt(a.clone(), la);
            let bv = apply_opt(b.clone(), ra);
            compare_for_eq(&av, &bv, meta.collation)
        }
    }
}

/// A binary comparison (`< <= > >= = != IS IS NOT`). Affinity is applied to each
/// operand by move; `null_safe` selects `IS`/`IS NOT`, which never yield NULL.
fn compare_op(op: CmpOp, null_safe: bool, lv: Value, rv: Value, meta: &CompareMeta) -> Value {
    let lv = apply_opt(lv, meta.apply_left);
    let rv = apply_opt(rv, meta.apply_right);

    if null_safe {
        // IS / IS NOT: NULL is just another comparable value here.
        let equal = match (lv.is_null(), rv.is_null()) {
            (true, true) => true,
            (true, false) | (false, true) => false,
            (false, false) => compare_for_eq(&lv, &rv, meta.collation) == Some(Ordering::Equal),
        };
        let out = if matches!(op, CmpOp::Ne) { !equal } else { equal };
        return Value::Integer(out as i64);
    }

    match compare_for_eq(&lv, &rv, meta.collation) {
        None => Value::Null, // a NULL operand makes an ordinary comparison unknown
        Some(ord) => {
            let t = match op {
                CmpOp::Lt => ord == Ordering::Less,
                CmpOp::Le => ord != Ordering::Greater,
                CmpOp::Gt => ord == Ordering::Greater,
                CmpOp::Ge => ord != Ordering::Less,
                CmpOp::Eq => ord == Ordering::Equal,
                CmpOp::Ne => ord != Ordering::Equal,
            };
            Value::Integer(t as i64)
        }
    }
}

/// Arithmetic (`+ - * / %`). NULL if either operand is NULL. `+ - * /` coerce via
/// [`as_num`] and stay in the integer domain when both operands are integers
/// (promoting to REAL only on overflow), else compute in `f64`. `%` computes its
/// remainder on both operands truncated to i64 but takes its result storage class
/// from the operand types just like the others (INTEGER only when both operands are
/// integers, else REAL). Division and modulo by zero yield NULL.
fn arith_op(op: ArithOp, l: Value, r: Value) -> Value {
    if l.is_null() || r.is_null() {
        return Value::Null;
    }
    // `%` computes an INTEGER remainder on both operands truncated toward zero to
    // i64, but its RESULT storage class follows the operand types exactly like
    // `+ - * /`: INTEGER only when both operands coerce to integers, else REAL
    // (datatype3.html §5 singles `%` out — it "returns either INTEGER or REAL (or
    // NULL) depending on the type of its operands"). This is unlike the bitwise ops
    // `& | << >>`, which truncate reals the same way but ALWAYS return INTEGER.
    //
    // Value and type read text operands at different widths: the remainder value
    // uses `to_integer`'s longest *integer* prefix, while the INT-vs-REAL decision
    // uses `as_num`'s longest *numeric* prefix (which may carry a fraction/exponent
    // and stay REAL). So `'2abc' % 3` is INTEGER 2, but `'2.5abc' % 2` computes on
    // the truncated 2 yet is REAL (its numeric prefix `2.5` is real-form).
    //
    // Divide-by-zero is decided on the truncated divisor and yields NULL on both the
    // integer and the real path (`5 % 0` and `5.5 % 0.4` are both NULL, since
    // truncating 0.4 gives 0), so the zero check precedes the type decision.
    if matches!(op, ArithOp::Mod) {
        let d = to_integer(&r);
        if d == 0 {
            return Value::Null;
        }
        // wrapping_rem is exact for every non-zero divisor and avoids the
        // i64::MIN % -1 overflow (which is 0 mathematically anyway).
        let rem = to_integer(&l).wrapping_rem(d);
        return match (as_num(&l), as_num(&r)) {
            (Num::Int(_), Num::Int(_)) => Value::Integer(rem),
            _ => Value::Real(rem as f64),
        };
    }

    match (as_num(&l), as_num(&r)) {
        (Num::Int(a), Num::Int(b)) => match op {
            ArithOp::Add => a
                .checked_add(b)
                .map(Value::Integer)
                .unwrap_or_else(|| Value::Real(a as f64 + b as f64)),
            ArithOp::Sub => a
                .checked_sub(b)
                .map(Value::Integer)
                .unwrap_or_else(|| Value::Real(a as f64 - b as f64)),
            ArithOp::Mul => a
                .checked_mul(b)
                .map(Value::Integer)
                .unwrap_or_else(|| Value::Real(a as f64 * b as f64)),
            ArithOp::Div => {
                if b == 0 {
                    Value::Null
                } else {
                    // checked_div is None only for i64::MIN / -1, which overflows
                    // i64; SQLite promotes that one case to REAL.
                    a.checked_div(b)
                        .map(Value::Integer)
                        .unwrap_or_else(|| Value::Real(a as f64 / b as f64))
                }
            }
            ArithOp::Mod => {
                debug_assert!(false, "Mod is handled by the early return above");
                Value::Null
            }
        },
        (x, y) => {
            let (x, y) = (x.to_f64(), y.to_f64());
            match op {
                ArithOp::Add => Value::Real(x + y),
                ArithOp::Sub => Value::Real(x - y),
                ArithOp::Mul => Value::Real(x * y),
                ArithOp::Div => {
                    if y == 0.0 {
                        Value::Null
                    } else {
                        Value::Real(x / y)
                    }
                }
                ArithOp::Mod => {
                    debug_assert!(false, "Mod is handled by the early return above");
                    Value::Null
                }
            }
        }
    }
}

/// Unary prefix operators. `Pos` is a true identity (no coercion); `Neg` coerces
/// then negates (promoting i64::MIN to REAL); `Not` is three-valued; `BitNot`
/// complements the integer coercion.
fn unary(op: UnaryOp, v: Value) -> Value {
    match op {
        UnaryOp::Pos => v, // returned completely untouched, even a non-numeric text
        UnaryOp::Neg => {
            if v.is_null() {
                return Value::Null;
            }
            match as_num(&v) {
                Num::Int(i) => i
                    .checked_neg()
                    .map(Value::Integer)
                    .unwrap_or_else(|| Value::Real(-(i as f64))),
                Num::Real(r) => Value::Real(-r),
            }
        }
        UnaryOp::Not => match truth(&v) {
            None => Value::Null,
            Some(true) => Value::Integer(0),
            Some(false) => Value::Integer(1),
        },
        UnaryOp::BitNot => {
            if v.is_null() {
                Value::Null
            } else {
                Value::Integer(!to_integer(&v))
            }
        }
    }
}

/// Integer bitwise/shift operators. NULL if either operand is NULL; both operands
/// are truncated to i64 first.
fn bit_op(op: BitOp, l: Value, r: Value) -> Value {
    if l.is_null() || r.is_null() {
        return Value::Null;
    }
    let a = to_integer(&l);
    let b = to_integer(&r);
    Value::Integer(match op {
        BitOp::And => a & b,
        BitOp::Or => a | b,
        BitOp::Shl => shift_left(a, b),
        BitOp::Shr => shift_right(a, b),
    })
}

/// `a << b` with SQLite's semantics: a shift of 64 or more is 0; a negative count
/// shifts the other way; `>>` is arithmetic (sign-filling).
fn shift_left(a: i64, b: i64) -> i64 {
    if b >= 64 {
        0
    } else if b <= -64 {
        a >> 63 // sign fill: a >> 63 is -1 for negative a, 0 otherwise
    } else if b >= 0 {
        a.wrapping_shl(b as u32) // b in 0..64, so this is exactly a << b
    } else {
        a >> ((-b) as u32) // -b in 1..64
    }
}

/// `a >> b` (arithmetic) — the mirror of [`shift_left`].
fn shift_right(a: i64, b: i64) -> i64 {
    if b >= 64 {
        a >> 63
    } else if b <= -64 {
        0
    } else if b >= 0 {
        a >> (b as u32)
    } else {
        a.wrapping_shl((-b) as u32)
    }
}

/// `||` string concatenation: NULL if either side is NULL, else both sides are cast
/// to TEXT and joined.
fn concat(l: Value, r: Value) -> Value {
    if l.is_null() || r.is_null() {
        return Value::Null;
    }
    let mut s = as_text(l);
    s.push_str(&as_text(r));
    Value::Text(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::{
        AggregateAccumulator, AggregateFunction, ScalarFunction,
    };
    use crate::ir::{CaseWhen, SubqueryId};
    use minisqlite_types::Collation;
    use std::sync::Arc;

    // ----- a stub evaluation context -----------------------------------------

    struct TestCtx {
        params: Vec<Value>,
        now_ms: i64,
        rng: u64,
        scalar_sub: Value,
        exists: bool,
        in_result: Option<bool>,
        in_row_result: Option<bool>,
        raise_ignored: bool,
    }

    impl Default for TestCtx {
        fn default() -> Self {
            TestCtx {
                params: Vec::new(),
                now_ms: 1_234_567_890_000,
                rng: 0x9E3779B97F4A7C15,
                scalar_sub: Value::Null,
                exists: false,
                in_result: Some(false),
                in_row_result: Some(false),
                raise_ignored: false,
            }
        }
    }

    impl FnContext for TestCtx {
        fn now_unix_millis(&self) -> i64 {
            self.now_ms
        }
        fn random_i64(&mut self) -> i64 {
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.rng as i64
        }
        fn fill_random(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.random_i64() as u8;
            }
        }
        fn last_insert_rowid(&self) -> i64 {
            0
        }
        fn changes(&self) -> i64 {
            0
        }
        fn total_changes(&self) -> i64 {
            0
        }
    }

    impl EvalContext for TestCtx {
        fn param(&self, index: usize) -> Result<Value> {
            self.params.get(index).cloned().ok_or_else(|| Error::sql("param out of range"))
        }
        fn signal_raise_ignore(&mut self) {
            self.raise_ignored = true;
        }
        fn eval_scalar_subquery(&mut self, _id: SubqueryId, _regs: &[Value]) -> Result<Value> {
            Ok(self.scalar_sub.clone())
        }
        // Echo the requested column index so the `ScalarSubqueryColumn` dispatch test can
        // prove `col` is threaded through the evaluator to the callback (col 0 must NOT
        // collapse to the scalar path).
        fn eval_scalar_subquery_column(
            &mut self,
            _id: SubqueryId,
            col: usize,
            _regs: &[Value],
        ) -> Result<Value> {
            Ok(Value::Integer(col as i64))
        }
        fn eval_exists(&mut self, _id: SubqueryId, _regs: &[Value]) -> Result<bool> {
            Ok(self.exists)
        }
        fn eval_in_subquery(
            &mut self,
            _id: SubqueryId,
            _probe: &Value,
            _meta: &CompareMeta,
            _regs: &[Value],
        ) -> Result<Option<bool>> {
            Ok(self.in_result)
        }
        fn eval_in_subquery_row(
            &mut self,
            _id: SubqueryId,
            _probe: &[Value],
            _metas: &[CompareMeta],
            _regs: &[Value],
        ) -> Result<Option<bool>> {
            Ok(self.in_row_result)
        }
    }

    // ----- stub functions (prove the calling contract) -----------------------

    #[derive(Debug)]
    struct AddOne;
    impl ScalarFunction for AddOne {
        fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
            match args.first() {
                None | Some(Value::Null) => Ok(Value::Null), // its own NULL behavior
                Some(v) => Ok(Value::Integer(to_integer(v) + 1)),
            }
        }
    }

    #[derive(Debug)]
    struct NowSeconds;
    impl ScalarFunction for NowSeconds {
        fn call(&self, _args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
            Ok(Value::Integer(ctx.now_unix_millis() / 1000))
        }
    }

    #[derive(Debug)]
    struct SumAgg;
    impl AggregateFunction for SumAgg {
        fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
            Box::new(SumAcc(0))
        }
    }
    struct SumAcc(i64);
    impl AggregateAccumulator for SumAcc {
        fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
            if let Some(v) = args.first() {
                self.0 += to_integer(v);
            }
            Ok(())
        }
        fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
            Ok(Value::Integer(self.0))
        }
    }

    // ----- helpers -----------------------------------------------------------

    fn int(i: i64) -> Value {
        Value::Integer(i)
    }
    fn real(r: f64) -> Value {
        Value::Real(r)
    }
    fn txt(s: &str) -> Value {
        Value::Text(s.into())
    }
    fn lit(v: Value) -> EvalExpr {
        EvalExpr::Literal(v)
    }
    fn bx(e: EvalExpr) -> Box<EvalExpr> {
        Box::new(e)
    }
    fn meta() -> CompareMeta {
        CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
    }
    fn meta_aff(l: Option<Affinity>, r: Option<Affinity>, c: Collation) -> CompareMeta {
        CompareMeta { apply_left: l, apply_right: r, collation: c }
    }

    fn veq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Integer(x), Value::Integer(y)) => x == y,
            (Value::Real(x), Value::Real(y)) => x == y || (x.is_nan() && y.is_nan()),
            (Value::Text(x), Value::Text(y)) => x == y,
            (Value::Blob(x), Value::Blob(y)) => x == y,
            _ => false,
        }
    }

    fn ev(e: &EvalExpr) -> Value {
        let mut ctx = TestCtx::default();
        eval(e, &[], &mut ctx).expect("eval should not error")
    }

    fn ev_regs(e: &EvalExpr, regs: &[Value]) -> Value {
        let mut ctx = TestCtx::default();
        eval(e, regs, &mut ctx).expect("eval should not error")
    }

    fn check(e: EvalExpr, want: Value) {
        let got = ev(&e);
        assert!(veq(&got, &want), "eval({e:?}) = {got:?}, want {want:?}");
    }

    fn arith(op: ArithOp, l: Value, r: Value) -> EvalExpr {
        EvalExpr::Arith { op, left: bx(lit(l)), right: bx(lit(r)) }
    }
    fn cmp(op: CmpOp, l: Value, r: Value, m: CompareMeta) -> EvalExpr {
        EvalExpr::Compare { op, null_safe: false, left: bx(lit(l)), right: bx(lit(r)), meta: m }
    }

    // ----- arithmetic --------------------------------------------------------

    #[test]
    fn arithmetic_examples() {
        check(arith(ArithOp::Div, int(5), int(2)), int(2)); // integer division
        check(arith(ArithOp::Div, real(5.0), int(2)), real(2.5)); // real path
        check(arith(ArithOp::Div, int(5), int(0)), Value::Null); // /0 -> NULL
        check(arith(ArithOp::Mod, int(5), int(0)), Value::Null); // %0 -> NULL
        check(arith(ArithOp::Mod, int(5), int(3)), int(2));
        check(arith(ArithOp::Add, int(2), int(3)), int(5));
        check(arith(ArithOp::Add, txt("2"), int(3)), int(5)); // '2' -> int
        check(arith(ArithOp::Add, txt("2.0"), int(3)), real(5.0)); // '2.0' -> real
        check(arith(ArithOp::Add, txt("abc"), int(3)), int(3)); // non-numeric -> 0
        check(arith(ArithOp::Add, txt(""), int(5)), int(5));
    }

    #[test]
    fn modulo_sign_follows_dividend() {
        check(arith(ArithOp::Mod, int(-5), int(3)), int(-2)); // C semantics
        check(arith(ArithOp::Mod, int(5), int(-3)), int(2));
        // i64::MIN % -1 must not overflow; it is 0.
        check(arith(ArithOp::Mod, int(i64::MIN), int(-1)), int(0));
    }

    #[test]
    fn modulo_result_type_follows_operands() {
        // The remainder is always the same integer computation on truncated
        // operands, but the RESULT storage class is REAL whenever either operand is
        // REAL (datatype3 §5), unlike the bitwise ops which always stay INTEGER.
        check(arith(ArithOp::Mod, real(5.5), real(2.0)), real(1.0)); // both real
        check(arith(ArithOp::Mod, real(5.5), int(2)), real(1.0)); // real dividend
        check(arith(ArithOp::Mod, int(2), real(5.5)), real(2.0)); // real divisor
        check(arith(ArithOp::Mod, int(7), int(3)), int(1)); // both int -> INTEGER
        // Divide-by-zero on the truncated divisor is NULL on BOTH the integer and
        // the real path (0.4 truncates to 0), decided before the type choice.
        check(arith(ArithOp::Mod, int(5), int(0)), Value::Null);
        check(arith(ArithOp::Mod, real(5.5), real(0.4)), Value::Null);
        // Sign follows the dividend; an all-integer result stays INTEGER.
        check(arith(ArithOp::Mod, int(-7), int(3)), int(-1));
    }

    #[test]
    fn modulo_text_operand_value_and_type_use_different_prefix_widths() {
        // A text operand's remainder VALUE uses its longest INTEGER prefix
        // (`to_integer`), while the result TYPE uses its longest NUMERIC prefix
        // (`as_num`) — the same operand-type test `+ - * /` use. Those widths diverge
        // when the numeric prefix carries a fraction/exponent: `'2.5abc'` truncates to
        // 2 for the computation (2 % 2 = 0), yet its numeric prefix `2.5` is real-form,
        // so the RESULT is Real(0.0). A plain integer-prefix text stays INTEGER. This
        // guards the `as_num`-not-`to_integer` type decision: a variant deriving the
        // type from `to_integer` would wrongly return Integer here.
        check(arith(ArithOp::Mod, txt("2.5abc"), int(2)), real(0.0));
        check(arith(ArithOp::Mod, txt("2abc"), int(3)), int(2));
    }

    #[test]
    fn arithmetic_overflow_promotes_to_real() {
        let e = arith(ArithOp::Add, int(i64::MAX), int(1));
        match ev(&e) {
            Value::Real(r) => assert!(r > 9.2e18, "got {r}"),
            other => panic!("expected REAL, got {other:?}"),
        }
        // i64::MIN / -1 overflows i64 and promotes to REAL.
        match ev(&arith(ArithOp::Div, int(i64::MIN), int(-1))) {
            Value::Real(r) => assert!(r > 9.2e18, "got {r}"),
            other => panic!("expected REAL, got {other:?}"),
        }
    }

    #[test]
    fn arithmetic_null_propagates() {
        check(arith(ArithOp::Add, Value::Null, int(3)), Value::Null);
        check(arith(ArithOp::Mul, int(3), Value::Null), Value::Null);
        check(arith(ArithOp::Mod, Value::Null, int(3)), Value::Null);
    }

    // ----- unary -------------------------------------------------------------

    #[test]
    fn unary_operators() {
        // Neg
        check(EvalExpr::Unary { op: UnaryOp::Neg, operand: bx(lit(int(5))) }, int(-5));
        check(EvalExpr::Unary { op: UnaryOp::Neg, operand: bx(lit(txt("abc"))) }, int(0));
        check(EvalExpr::Unary { op: UnaryOp::Neg, operand: bx(lit(Value::Null)) }, Value::Null);
        match ev(&EvalExpr::Unary { op: UnaryOp::Neg, operand: bx(lit(int(i64::MIN))) }) {
            Value::Real(r) => assert!(r > 9.2e18),
            other => panic!("expected REAL for -i64::MIN, got {other:?}"),
        }
        // Pos returns operand completely untouched (no coercion).
        check(EvalExpr::Unary { op: UnaryOp::Pos, operand: bx(lit(txt("abc"))) }, txt("abc"));
        check(EvalExpr::Unary { op: UnaryOp::Pos, operand: bx(lit(Value::Null)) }, Value::Null);
        // BitNot
        check(EvalExpr::Unary { op: UnaryOp::BitNot, operand: bx(lit(int(0))) }, int(-1));
        check(EvalExpr::Unary { op: UnaryOp::BitNot, operand: bx(lit(Value::Null)) }, Value::Null);
    }

    #[test]
    fn not_is_three_valued() {
        check(EvalExpr::Unary { op: UnaryOp::Not, operand: bx(lit(int(0))) }, int(1));
        check(EvalExpr::Unary { op: UnaryOp::Not, operand: bx(lit(int(5))) }, int(0));
        check(EvalExpr::Unary { op: UnaryOp::Not, operand: bx(lit(Value::Null)) }, Value::Null);
    }

    // ----- bitwise/shift -----------------------------------------------------

    #[test]
    fn bitwise_and_shifts() {
        let bit = |op, l, r| EvalExpr::Bitwise { op, left: bx(lit(l)), right: bx(lit(r)) };
        check(bit(BitOp::Shl, int(1), int(2)), int(4));
        check(bit(BitOp::Shr, int(8), int(1)), int(4));
        check(bit(BitOp::Shl, int(1), int(64)), int(0)); // >= 64 -> 0
        check(bit(BitOp::Shl, int(1), int(-1)), int(0)); // 1 << -1 == 1 >> 1 == 0
        check(bit(BitOp::Shr, int(1), int(1)), int(0));
        check(bit(BitOp::And, int(6), int(3)), int(2));
        check(bit(BitOp::Or, int(4), int(1)), int(5));
        check(bit(BitOp::Shl, int(1), int(63)), int(i64::MIN)); // bit 63 set
        check(bit(BitOp::Shr, int(-8), int(1)), int(-4)); // arithmetic
        check(bit(BitOp::Shr, int(-1), int(100)), int(-1)); // >=64 sign fill
        check(bit(BitOp::And, Value::Null, int(3)), Value::Null);
    }

    #[test]
    fn shift_negative_and_oversized_counts() {
        // Exercises the direction-reversal and magnitude>=64 arms of shift_left /
        // shift_right that `bitwise_and_shifts` doesn't reach — a sign/direction bug
        // in the mirrored `>>`/`<<` negative-count branches would show up here.
        let bit = |op, l, r| EvalExpr::Bitwise { op, left: bx(lit(l)), right: bx(lit(r)) };
        // A small negative count reverses direction.
        check(bit(BitOp::Shr, int(8), int(-1)), int(16)); // 8 >> -1 == 8 << 1
        check(bit(BitOp::Shr, int(-3), int(-2)), int(-12)); // >> negative -> wrapping_shl
        check(bit(BitOp::Shl, int(-4), int(-1)), int(-2)); // -4 << -1 == -4 >> 1 (arithmetic)
        // Magnitude >= 64: `<<` sign-fills via a>>63; the reversed `>>` yields 0.
        check(bit(BitOp::Shl, int(-1), int(-64)), int(-1)); // -1 << -64 == -1 >> 63
        check(bit(BitOp::Shl, int(8), int(-64)), int(0)); // 8 << -64 == 8 >> 63 == 0
        check(bit(BitOp::Shr, int(1), int(-64)), int(0)); // 1 >> -64 == 1 << 64 == 0
        check(bit(BitOp::Shr, int(-1), int(-64)), int(0)); // -1 >> -64 == -1 << 64 == 0
    }

    // ----- 3VL AND/OR --------------------------------------------------------

    #[test]
    fn and_or_truth_tables_with_null() {
        let and = |l, r| EvalExpr::And(bx(lit(l)), bx(lit(r)));
        let or = |l, r| EvalExpr::Or(bx(lit(l)), bx(lit(r)));
        // AND
        check(and(int(1), int(1)), int(1));
        check(and(int(1), int(0)), int(0));
        check(and(int(0), Value::Null), int(0)); // false short-circuits
        check(and(int(1), Value::Null), Value::Null);
        check(and(Value::Null, int(1)), Value::Null);
        check(and(Value::Null, int(0)), int(0));
        check(and(Value::Null, Value::Null), Value::Null);
        // OR
        check(or(int(0), int(0)), int(0));
        check(or(int(1), Value::Null), int(1)); // true short-circuits
        check(or(Value::Null, int(1)), int(1));
        check(or(int(0), Value::Null), Value::Null);
        check(or(Value::Null, Value::Null), Value::Null);
    }

    // ----- comparison + affinity + collation ---------------------------------

    #[test]
    fn comparison_basic_and_null() {
        check(cmp(CmpOp::Lt, int(1), int(2), meta()), int(1));
        check(cmp(CmpOp::Ge, int(2), int(2), meta()), int(1));
        check(cmp(CmpOp::Eq, int(2), int(3), meta()), int(0));
        check(cmp(CmpOp::Ne, int(2), int(3), meta()), int(1));
        // NULL operand -> NULL for ordinary comparison.
        check(cmp(CmpOp::Eq, Value::Null, int(3), meta()), Value::Null);
        check(cmp(CmpOp::Lt, int(3), Value::Null, meta()), Value::Null);
    }

    #[test]
    fn comparison_cross_class_number_vs_text() {
        // Without affinity, a number is always less than text (5 < '5').
        check(cmp(CmpOp::Lt, int(5), txt("5"), meta()), int(1));
        check(cmp(CmpOp::Eq, int(5), txt("5"), meta()), int(0));
        // With NUMERIC affinity applied to the text side, '5' becomes 5 and they
        // compare equal.
        let m = meta_aff(None, Some(Affinity::Numeric), Collation::Binary);
        check(cmp(CmpOp::Eq, int(5), txt("5"), m), int(1));
    }

    #[test]
    fn comparison_uses_collation() {
        // 'abc' = 'ABC' is false under BINARY, true under NOCASE.
        check(cmp(CmpOp::Eq, txt("abc"), txt("ABC"), meta()), int(0));
        let nocase = meta_aff(None, None, Collation::NoCase);
        check(cmp(CmpOp::Eq, txt("abc"), txt("ABC"), nocase), int(1));
    }

    #[test]
    fn is_and_is_not_are_null_safe() {
        let is = |l, r| EvalExpr::Compare {
            op: CmpOp::Eq,
            null_safe: true,
            left: bx(lit(l)),
            right: bx(lit(r)),
            meta: meta(),
        };
        let isnot = |l, r| EvalExpr::Compare {
            op: CmpOp::Ne,
            null_safe: true,
            left: bx(lit(l)),
            right: bx(lit(r)),
            meta: meta(),
        };
        check(is(Value::Null, Value::Null), int(1)); // NULL IS NULL -> 1
        check(is(int(1), Value::Null), int(0)); // 1 IS NULL -> 0
        check(is(int(1), int(1)), int(1));
        check(isnot(Value::Null, Value::Null), int(0)); // NULL IS NOT NULL -> 0
        check(isnot(int(1), Value::Null), int(1));
        check(isnot(int(1), int(2)), int(1));
    }

    #[test]
    fn is_null_and_not_null_nodes() {
        // The dedicated ISNULL/NOTNULL nodes are a different code path from the
        // null-safe Compare tested above; pin each so a swapped 1/0 can't slip.
        check(EvalExpr::IsNull(bx(lit(Value::Null))), int(1));
        check(EvalExpr::IsNull(bx(lit(int(0)))), int(0)); // 0 is not NULL
        check(EvalExpr::IsNull(bx(lit(txt("")))), int(0)); // empty text is not NULL
        check(EvalExpr::NotNull(bx(lit(Value::Null))), int(0));
        check(EvalExpr::NotNull(bx(lit(int(0)))), int(1));
    }

    // ----- IN list -----------------------------------------------------------

    #[test]
    fn in_list_membership() {
        let in_list = |subj: Value, items: Vec<Value>, negated| EvalExpr::InList {
            negated,
            subject: bx(lit(subj)),
            items: items.into_iter().map(lit).collect(),
            meta: meta(),
        };
        check(in_list(int(2), vec![int(1), int(2), int(3)], false), int(1));
        check(in_list(int(9), vec![int(1), int(2), int(3)], false), int(0));
        check(in_list(int(9), vec![int(1), int(2)], true), int(1)); // NOT IN
        // Empty list: IN () is 0, NOT IN () is 1 — even for NULL subject.
        check(in_list(int(1), vec![], false), int(0));
        check(in_list(int(1), vec![], true), int(1));
        check(in_list(Value::Null, vec![], false), int(0));
        // NULL handling: not found + a NULL in the list -> unknown (NULL).
        check(in_list(int(9), vec![int(1), Value::Null], false), Value::Null);
        check(in_list(int(1), vec![int(1), Value::Null], false), int(1)); // found wins over NULL
        check(in_list(Value::Null, vec![int(1)], false), Value::Null); // NULL probe -> NULL
    }

    #[test]
    fn in_list_applies_affinity() {
        // 2 IN ('1','2','3') with NUMERIC affinity on the items matches.
        let e = EvalExpr::InList {
            negated: false,
            subject: bx(lit(int(2))),
            items: vec![lit(txt("1")), lit(txt("2")), lit(txt("3"))],
            meta: meta_aff(None, Some(Affinity::Numeric), Collation::Binary),
        };
        check(e, int(1));
    }

    // ----- BETWEEN -----------------------------------------------------------

    #[test]
    fn between_inclusive_and_null() {
        let bt = |v: Value, lo: Value, hi: Value, negated| EvalExpr::Between {
            negated,
            subject: bx(lit(v)),
            low: bx(lit(lo)),
            high: bx(lit(hi)),
            low_meta: meta(),
            high_meta: meta(),
        };
        check(bt(int(5), int(1), int(10), false), int(1));
        check(bt(int(1), int(1), int(10), false), int(1)); // inclusive low
        check(bt(int(10), int(1), int(10), false), int(1)); // inclusive high
        check(bt(int(0), int(1), int(10), false), int(0));
        check(bt(int(0), int(1), int(10), true), int(1)); // NOT BETWEEN
        // NULL bound -> unknown where the other bound doesn't already decide it.
        check(bt(int(5), Value::Null, int(10), false), Value::Null);
        check(bt(int(50), int(1), Value::Null, false), Value::Null);
        // A definite false from one side wins over a NULL on the other.
        check(bt(int(50), int(1), int(10), false), int(0));
    }

    // ----- CASE --------------------------------------------------------------

    #[test]
    fn case_searched() {
        // CASE WHEN 0 THEN 'a' WHEN 1 THEN 'b' ELSE 'c' END -> 'b'
        let e = EvalExpr::Case {
            operand: None,
            whens: vec![
                CaseWhen { when: lit(int(0)), cmp: None, then: lit(txt("a")) },
                CaseWhen { when: lit(int(1)), cmp: None, then: lit(txt("b")) },
            ],
            else_expr: Some(bx(lit(txt("c")))),
        };
        check(e, txt("b"));
        // No arm matches, no ELSE -> NULL.
        let e2 = EvalExpr::Case {
            operand: None,
            whens: vec![CaseWhen { when: lit(int(0)), cmp: None, then: lit(txt("a")) }],
            else_expr: None,
        };
        check(e2, Value::Null);
    }

    #[test]
    fn case_simple_null_does_not_match_null() {
        // CASE NULL WHEN NULL THEN 1 ELSE 2 END -> 2 (NULL never matches NULL).
        let e = EvalExpr::Case {
            operand: Some(bx(lit(Value::Null))),
            whens: vec![CaseWhen { when: lit(Value::Null), cmp: Some(meta()), then: lit(int(1)) }],
            else_expr: Some(bx(lit(int(2)))),
        };
        check(e, int(2));
        // CASE 2 WHEN 1 THEN 'a' WHEN 2 THEN 'b' END -> 'b'
        let e2 = EvalExpr::Case {
            operand: Some(bx(lit(int(2)))),
            whens: vec![
                CaseWhen { when: lit(int(1)), cmp: Some(meta()), then: lit(txt("a")) },
                CaseWhen { when: lit(int(2)), cmp: Some(meta()), then: lit(txt("b")) },
            ],
            else_expr: None,
        };
        check(e2, txt("b"));
    }

    // ----- RAISE -------------------------------------------------------------

    #[test]
    fn raise_abort_fail_rollback_yield_constraint_error_with_message() {
        // RAISE(ABORT|FAIL|ROLLBACK, msg) evaluates to a constraint error carrying the
        // message verbatim; the firing DML operator turns the propagated error into a
        // statement abort (which the engine's implicit transaction rolls back).
        for kind in [RaiseKind::Abort, RaiseKind::Fail, RaiseKind::Rollback] {
            let e = EvalExpr::Raise { kind, message: Some("nope".to_string()) };
            let mut ctx = TestCtx::default();
            let err = eval(&e, &[], &mut ctx).expect_err("RAISE must error");
            assert!(matches!(err, Error::Constraint(ref m) if m == "nope"), "got {err:?}");
        }
    }

    #[test]
    fn raise_ignore_signals_the_context_and_returns_a_sentinel_error() {
        // RAISE(IGNORE) is a non-error row-skip signal the pinned `Error` cannot carry, so
        // the evaluator (1) calls `signal_raise_ignore` on the context — the executor turns
        // this into a row-skip — and (2) returns a sentinel `Err` to unwind out of the
        // trigger body. A context that does not act on the signal (this mock, a bare
        // `SELECT RAISE(IGNORE)`) simply sees the error, which is the correct outcome where
        // there is no row to skip.
        let e = EvalExpr::Raise { kind: RaiseKind::Ignore, message: None };
        let mut ctx = TestCtx::default();
        assert!(!ctx.raise_ignored, "signal not yet sent");
        assert!(eval(&e, &[], &mut ctx).is_err(), "unwinds with a sentinel error");
        assert!(ctx.raise_ignored, "the IGNORE control signal reached the context");
    }

    // ----- CAST / concat / coalesce / nullif ---------------------------------

    #[test]
    fn cast_operator() {
        let cast = |v: Value, a| EvalExpr::Cast { affinity: a, operand: bx(lit(v)) };
        check(cast(real(3.9), Affinity::Integer), int(3)); // truncates
        check(cast(txt("42abc"), Affinity::Integer), int(42)); // prefix
        check(cast(int(65), Affinity::Text), txt("65"));
        check(cast(Value::Null, Affinity::Integer), Value::Null);
    }

    #[test]
    fn concat_operator() {
        let cc = |l: Value, r: Value| EvalExpr::Concat { left: bx(lit(l)), right: bx(lit(r)) };
        check(cc(int(1), int(2)), txt("12"));
        check(cc(real(1.5), txt("x")), txt("1.5x"));
        check(cc(txt("a"), Value::Null), Value::Null); // NULL propagates
        check(cc(Value::Null, txt("a")), Value::Null);
    }

    #[test]
    fn coalesce_and_nullif() {
        let coalesce = |items: Vec<Value>| EvalExpr::Coalesce(items.into_iter().map(lit).collect());
        check(coalesce(vec![Value::Null, Value::Null, int(3), int(4)]), int(3));
        check(coalesce(vec![Value::Null, Value::Null]), Value::Null);

        let nullif = |l: Value, r: Value| EvalExpr::NullIf {
            left: bx(lit(l)),
            right: bx(lit(r)),
            meta: meta(),
        };
        check(nullif(int(1), int(1)), Value::Null);
        check(nullif(int(1), int(2)), int(1)); // returns the ORIGINAL left
        check(nullif(Value::Null, int(1)), Value::Null);
    }

    // ----- LIKE / GLOB through the eval node ----------------------------------

    #[test]
    fn like_and_glob_nodes() {
        let like = |s: Value, p: Value, negated| EvalExpr::Like {
            negated,
            kind: LikeKind::Like,
            subject: bx(lit(s)),
            pattern: bx(lit(p)),
            escape: None,
        };
        check(like(txt("Hello"), txt("h%o"), false), int(1)); // case-insensitive
        check(like(txt("Hello"), txt("h%o"), true), int(0)); // NOT LIKE
        check(like(Value::Null, txt("%"), false), Value::Null); // NULL subject
        check(like(txt("x"), Value::Null, false), Value::Null); // NULL pattern
        // Numbers are cast to text: 5 LIKE '5' -> 1.
        check(like(int(5), txt("5"), false), int(1));

        let glob = |s: Value, p: Value| EvalExpr::Like {
            negated: false,
            kind: LikeKind::Glob,
            subject: bx(lit(s)),
            pattern: bx(lit(p)),
            escape: None,
        };
        check(glob(txt("abc"), txt("a[b-d]c")), int(1));
        check(glob(txt("ABC"), txt("abc")), int(0)); // case-sensitive
    }

    #[test]
    fn like_with_escape() {
        let like_esc = |s: &str, p: &str, esc: Value| EvalExpr::Like {
            negated: false,
            kind: LikeKind::Like,
            subject: bx(lit(txt(s))),
            pattern: bx(lit(txt(p))),
            escape: Some(bx(lit(esc))),
        };
        check(like_esc("50%", "50\\%", txt("\\")), int(1));
        check(like_esc("505", "50\\%", txt("\\")), int(0)); // \% is a literal %
        // NULL escape -> NULL.
        check(like_esc("50%", "50\\%", Value::Null), Value::Null);
    }

    #[test]
    fn like_escape_must_be_single_char() {
        let mut ctx = TestCtx::default();
        let e = EvalExpr::Like {
            negated: false,
            kind: LikeKind::Like,
            subject: bx(lit(txt("x"))),
            pattern: bx(lit(txt("x"))),
            escape: Some(bx(lit(txt("ab")))), // two chars -> error
        };
        assert!(eval(&e, &[], &mut ctx).is_err());
    }

    // ----- columns, params, functions, subqueries, now -----------------------

    #[test]
    fn column_reads_register() {
        check_regs(EvalExpr::Column(1), &[int(10), int(20), int(30)], int(20));
    }

    fn check_regs(e: EvalExpr, regs: &[Value], want: Value) {
        let got = ev_regs(&e, regs);
        assert!(veq(&got, &want), "eval({e:?}) = {got:?}, want {want:?}");
    }

    #[test]
    fn column_out_of_range_errors() {
        let mut ctx = TestCtx::default();
        assert!(eval(&EvalExpr::Column(5), &[int(1)], &mut ctx).is_err());
    }

    #[test]
    fn param_reads_and_errors() {
        let mut ctx = TestCtx { params: vec![int(7), txt("hi")], ..TestCtx::default() };
        assert!(veq(&eval(&EvalExpr::Param(0), &[], &mut ctx).unwrap(), &int(7)));
        assert!(veq(&eval(&EvalExpr::Param(1), &[], &mut ctx).unwrap(), &txt("hi")));
        assert!(eval(&EvalExpr::Param(9), &[], &mut ctx).is_err());
    }

    #[test]
    fn scalar_function_call() {
        // AddOne(9) -> 10; AddOne(NULL) -> NULL (function decides).
        let f: Arc<dyn ScalarFunction> = Arc::new(AddOne);
        check(EvalExpr::Func { func: f.clone(), args: vec![lit(int(9))] }, int(10));
        check(EvalExpr::Func { func: f, args: vec![lit(Value::Null)] }, Value::Null);
    }

    #[test]
    fn function_receives_context() {
        // NowSeconds reads now_unix_millis through the upcast FnContext.
        let f: Arc<dyn ScalarFunction> = Arc::new(NowSeconds);
        check(EvalExpr::Func { func: f, args: vec![] }, int(1_234_567_890));
    }

    #[test]
    fn aggregate_accumulator_folds() {
        // Prove the aggregate traits are usable: sum a few values.
        let agg = SumAgg;
        let mut acc = agg.new_accumulator(Collation::Binary);
        let mut ctx = TestCtx::default();
        for v in [int(1), int(2), int(3)] {
            acc.step(&[v], &mut ctx).unwrap();
        }
        assert!(veq(&acc.finalize(&mut ctx).unwrap(), &int(6)));
    }

    #[test]
    fn subquery_callbacks() {
        // EXISTS / NOT EXISTS
        let mut ctx = TestCtx { exists: true, ..TestCtx::default() };
        assert!(veq(&eval(&EvalExpr::Exists { negated: false, id: 0 }, &[], &mut ctx).unwrap(), &int(1)));
        assert!(veq(&eval(&EvalExpr::Exists { negated: true, id: 0 }, &[], &mut ctx).unwrap(), &int(0)));

        // scalar subquery returns the ctx-provided value
        let mut ctx2 = TestCtx { scalar_sub: int(42), ..TestCtx::default() };
        assert!(veq(&eval(&EvalExpr::ScalarSubquery(0), &[], &mut ctx2).unwrap(), &int(42)));

        // ScalarSubqueryColumn dispatches to eval_scalar_subquery_column with `col`
        // threaded through (the TestCtx echoes the column index): col 0 and col 2 reach
        // the callback distinctly, so the variant is NOT silently folded to col 0.
        let mut ctx3 = TestCtx::default();
        assert!(veq(
            &eval(&EvalExpr::ScalarSubqueryColumn { id: 0, col: 0 }, &[], &mut ctx3).unwrap(),
            &int(0)
        ));
        assert!(veq(
            &eval(&EvalExpr::ScalarSubqueryColumn { id: 0, col: 2 }, &[], &mut ctx3).unwrap(),
            &int(2)
        ));

        // IN (subquery): three-valued, negation inverts Some but not None
        let in_sub = |negated| EvalExpr::InSubquery {
            negated,
            subject: bx(lit(int(1))),
            id: 0,
            meta: meta(),
        };
        let mut ctx3 = TestCtx { in_result: Some(true), ..TestCtx::default() };
        assert!(veq(&eval(&in_sub(false), &[], &mut ctx3).unwrap(), &int(1)));
        assert!(veq(&eval(&in_sub(true), &[], &mut ctx3).unwrap(), &int(0)));
        let mut ctx4 = TestCtx { in_result: None, ..TestCtx::default() };
        assert!(veq(&eval(&in_sub(false), &[], &mut ctx4).unwrap(), &Value::Null));
        assert!(veq(&eval(&in_sub(true), &[], &mut ctx4).unwrap(), &Value::Null)); // NULL stays NULL
    }

    #[test]
    fn in_subquery_row_lowers_three_valued_result_and_builds_the_probe() {
        // The row-value IN arm evaluates each subject element into the probe tuple, calls
        // `eval_in_subquery_row`, then applies the SAME negate-then-lower rule as the
        // scalar `InSubquery` (Some(true)->1, Some(false)->0, None->NULL; NOT flips Some
        // but leaves None). The context here returns a canned membership and IGNORES the
        // probe, so this pins ONLY the arm's lowering — not the tuple 3VL (which lives in
        // the executor) nor the probe's construction (covered end to end at exec, where
        // `correlated_in_subquery_row_reruns_per_outer_row` builds `(col(0), lit(2))`).
        let in_row = |negated| EvalExpr::InSubqueryRow {
            negated,
            subjects: vec![EvalExpr::Column(0), EvalExpr::Column(1)],
            id: 0,
            metas: vec![meta(), meta()],
        };
        let regs = [int(1), int(2)];

        let mut t = TestCtx { in_row_result: Some(true), ..TestCtx::default() };
        assert!(veq(&eval(&in_row(false), &regs, &mut t).unwrap(), &int(1)));
        assert!(veq(&eval(&in_row(true), &regs, &mut t).unwrap(), &int(0))); // NOT flips true

        let mut f = TestCtx { in_row_result: Some(false), ..TestCtx::default() };
        assert!(veq(&eval(&in_row(false), &regs, &mut f).unwrap(), &int(0)));
        assert!(veq(&eval(&in_row(true), &regs, &mut f).unwrap(), &int(1))); // NOT flips false

        let mut u = TestCtx { in_row_result: None, ..TestCtx::default() };
        assert!(veq(&eval(&in_row(false), &regs, &mut u).unwrap(), &Value::Null));
        assert!(veq(&eval(&in_row(true), &regs, &mut u).unwrap(), &Value::Null)); // NULL stays NULL
    }

    #[test]
    fn in_subquery_row_default_context_method_errors() {
        // The default `eval_in_subquery_row` (which non-executor mocks inherit) is a loud
        // error, not a silent wrong answer. A context that does NOT override it must make
        // an `InSubqueryRow` eval fail — the marker that only the real executor supports it.
        struct Bare;
        impl FnContext for Bare {
            fn now_unix_millis(&self) -> i64 {
                0
            }
            fn random_i64(&mut self) -> i64 {
                0
            }
            fn fill_random(&mut self, _buf: &mut [u8]) {}
            fn last_insert_rowid(&self) -> i64 {
                0
            }
            fn changes(&self) -> i64 {
                0
            }
            fn total_changes(&self) -> i64 {
                0
            }
        }
        impl EvalContext for Bare {
            fn param(&self, _index: usize) -> Result<Value> {
                Err(Error::sql("no params"))
            }
            fn eval_scalar_subquery(&mut self, _id: SubqueryId, _regs: &[Value]) -> Result<Value> {
                Err(Error::sql("no subqueries"))
            }
            fn eval_exists(&mut self, _id: SubqueryId, _regs: &[Value]) -> Result<bool> {
                Err(Error::sql("no subqueries"))
            }
            fn eval_in_subquery(
                &mut self,
                _id: SubqueryId,
                _probe: &Value,
                _meta: &CompareMeta,
                _regs: &[Value],
            ) -> Result<Option<bool>> {
                Err(Error::sql("no subqueries"))
            }
            // eval_in_subquery_row deliberately NOT overridden — uses the trait default.
        }
        let e = EvalExpr::InSubqueryRow {
            negated: false,
            subjects: vec![lit(int(1)), lit(int(2))],
            id: 0,
            metas: vec![meta(), meta()],
        };
        assert!(eval(&e, &[], &mut Bare).is_err(), "default row-IN context method must error");
    }

    #[test]
    fn current_timestamp_renders() {
        // now_ms default is 2009-02-13 23:31:30 UTC.
        check(EvalExpr::Now(crate::ir::NowKind::Timestamp), txt("2009-02-13 23:31:30"));
        check(EvalExpr::Now(crate::ir::NowKind::Date), txt("2009-02-13"));
        check(EvalExpr::Now(crate::ir::NowKind::Time), txt("23:31:30"));
    }

    #[test]
    fn collate_is_passthrough() {
        check(
            EvalExpr::Collate { collation: Collation::NoCase, operand: bx(lit(txt("Hi"))) },
            txt("Hi"),
        );
    }

    #[test]
    fn nested_expression_tree() {
        // (1 + 2) * 3 - 10 = -1, exercising recursion through the evaluator.
        let inner = EvalExpr::Arith {
            op: ArithOp::Mul,
            left: bx(EvalExpr::Arith {
                op: ArithOp::Add,
                left: bx(lit(int(1))),
                right: bx(lit(int(2))),
            }),
            right: bx(lit(int(3))),
        };
        let full = EvalExpr::Arith { op: ArithOp::Sub, left: bx(inner), right: bx(lit(int(10))) };
        check(full, int(-1));
    }
}
