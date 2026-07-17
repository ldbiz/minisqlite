//! Statement dispatch: the one place a parsed [`Statement`] is routed to its
//! handler. Transaction control, DDL, and `PRAGMA` are handled at the engine
//! (against the catalog and pager seams); `SELECT` and DML are compiled by the
//! planner and run by the executor. This module only routes and wraps
//! transactions — it never plans or executes itself.

use minisqlite_catalog::{Catalog, MultiCatalog};
use minisqlite_plan::{OnConflict, Plan, PlanNode, Planner};
use minisqlite_sql::{CreateTableBody, Statement};
use minisqlite_types::{Error, QueryResult, Result, Row};

use crate::engine::SqlEngine;

impl SqlEngine {
    /// Route one parsed statement to its handler, returning its result set if it
    /// produced one (a `SELECT` or a row-producing `PRAGMA`) or `None` otherwise.
    /// `source` is the verbatim statement text, used as the `sqlite_schema.sql`
    /// value for DDL.
    pub(crate) fn run_statement(
        &mut self,
        stmt: &Statement,
        source: &str,
    ) -> Result<Option<QueryResult>> {
        match stmt {
            // --- Transaction control: engine-level, never sent to the planner. ---
            Statement::Begin { mode } => {
                self.exec_begin(*mode)?;
                Ok(None)
            }
            Statement::Commit => {
                self.exec_commit()?;
                Ok(None)
            }
            Statement::Rollback { to_savepoint: None } => {
                self.exec_rollback()?;
                Ok(None)
            }
            // Savepoints: nested, named undo points within a transaction, handled at
            // the engine (name -> pager depth) over the pager's savepoint primitive.
            Statement::Savepoint(name) => {
                self.exec_savepoint(name)?;
                Ok(None)
            }
            Statement::Release(name) => {
                self.exec_release(name)?;
                Ok(None)
            }
            Statement::Rollback { to_savepoint: Some(name) } => {
                self.exec_rollback_to(name)?;
                Ok(None)
            }

            // --- PRAGMA: engine-level. ---
            Statement::Pragma { name, arg } => self.run_pragma(name, arg.as_ref()),

            // --- DDL the catalog seam can register/remove. ---
            Statement::CreateTable(ct) => match &ct.body {
                // CREATE TABLE ... AS SELECT is orchestrated by the engine (plan the
                // SELECT for its schema, register a plain table, run INSERT ... SELECT),
                // intercepted here BEFORE it reaches the catalog's `AsSelect` reject. A
                // plain `Columns` body goes straight to the catalog seam as before.
                CreateTableBody::AsSelect(sel) => self.exec_ctas(ct, sel, source),
                _ => {
                    // Route by the parsed `TEMP` flag / a `temp.`|`main.` qualifier; a
                    // temp create materializes the temp store first. Non-temp, unqualified
                    // → main (index 0), i.e. the pre-namespace path unchanged.
                    let db = self.create_target_db(ct.temp, &ct.name)?;
                    self.with_write_txn_in(db, |cat, pager| cat.create_table(pager, ct, source))?;
                    Ok(None)
                }
            },
            Statement::CreateIndex(ci) => {
                // `create_index` registers the (empty) index; the executor then
                // backfills it from the table's existing rows, INSIDE the same write
                // transaction so the schema row, the index b-tree, and its entries are
                // one atomic unit — a backfill error (e.g. a UNIQUE violation among the
                // pre-existing rows) rolls back the whole CREATE INDEX, leaving no index.
                // Backfill is skipped when `IF NOT EXISTS` finds the index already
                // present: `create_index` returns a plain `Ok` no-op for that case (and
                // errors, short-circuiting before the backfill, on any other collision
                // or a missing table), so a pre-existing index means the no-op path —
                // re-backfilling it would double-insert every entry.
                // Bind the target table's GENERATED-column programs AND the new index's
                // key EXPRESSIONS BEFORE opening the write transaction: an index over a
                // VIRTUAL generated column must compute that column to key its backfilled
                // entries, and an EXPRESSION index (`CREATE INDEX i ON t(a+b)`,
                // lang_createindex.html §1.2) must evaluate its key expression per row to
                // key them. Both bindings need the planner's function registry + a catalog
                // borrow the mut-borrowing write-txn closure cannot also take. The table
                // pre-exists (CREATE INDEX requires it), so they resolve now; each is empty
                // for a plain named-column index over a table with no generated column,
                // leaving that backfill unchanged.
                // An index lives in its table's namespace: a CREATE INDEX on a temp table
                // (or a `temp.`-qualified index name) registers and backfills in temp.
                // Resolve this FIRST so the generated/key bindings below are scoped to the
                // SAME namespace the index is built in — a schema-qualified
                // `CREATE INDEX main.i ON t` with a temp `t` shadowing must key its backfill
                // against main.t, not the search-order-first temp.t (`table_in(db, ..)`
                // below, not a bare `table(..)`). For an unqualified index the resolved db
                // equals the search-order owner, so this is behavior-identical there.
                let db = self.index_target_db(ci)?;
                // Resolve against a MultiCatalog (search order: temp, main, attached) but
                // look the target table up BY `db` so the binding matches the index's store;
                // its borrow ends here (owned `Vec`s returned) before the mut-borrowing
                // write closure.
                let mc = MultiCatalog::new(&self.catalogs, &self.namespaces);
                let generated = self.planner.table_generated_programs(&mc, db, &ci.table)?;
                let index_key_exprs = self.planner.index_key_programs(&mc, db, ci)?;
                // A PARTIAL `CREATE INDEX … WHERE p` binds its predicate too, so the backfill
                // indexes only the existing rows `p` admits (`partialindex.html` §2); a full
                // index binds `None` and backfills every row unchanged. Bound against the SAME
                // `db`-scoped table as the key programs above.
                let index_partial_predicate = self.planner.index_partial_predicate(&mc, db, ci)?;
                // A `temp.`-qualified CREATE INDEX before temp is materialized targets the
                // (empty, non-live) temp store. Run `create_index` against a throwaway empty
                // store so the concrete catalog yields the exact live-but-empty result — a
                // reserved-name error (`temp.sqlite_ix` → "object name reserved …", or
                // `ON sqlite_master` → "table sqlite_master may not be indexed") or "no such
                // table" — WITHOUT materializing temp. An empty store always errors before
                // backfill, so the programs bound above go unused on this cold path (they are
                // still bound identically to the live path via the MultiCatalog above, so a
                // bad expression key is rejected the same whether or not temp is live).
                if !self.namespace_live(db) {
                    self.with_absent_namespace(|cat, pager| cat.create_index(pager, ci, source))?;
                    return Ok(None);
                }
                self.with_write_txn_in(db, |cat, pager| {
                    let already_present = cat.index(&ci.name.name)?.is_some();
                    cat.create_index(pager, ci, source)?;
                    if !already_present {
                        minisqlite_exec::build_index(
                            &mut *pager,
                            &*cat,
                            &ci.name.name,
                            &generated,
                            &index_key_exprs,
                            index_partial_predicate.clone(),
                        )?;
                    }
                    Ok(())
                })?;
                Ok(None)
            }
            // CREATE VIEW / CREATE TRIGGER register a `type='view'` / `type='trigger'`
            // sqlite_schema row through the same catalog seam (rootpage 0 — neither owns
            // a b-tree — with the verbatim `source` stored as its `sql`). Only
            // registration is wired here; expanding a view inside a query and firing a
            // trigger on DML are separate planner/executor concerns, not this route's.
            Statement::CreateView(cv) => {
                let db = self.create_target_db(cv.temp, &cv.name)?;
                self.with_write_txn_in(db, |cat, pager| cat.create_view(pager, cv, source))?;
                Ok(None)
            }
            Statement::CreateTrigger(ct) => {
                // Resolve WHERE the trigger lives and WHERE its ON-target lives. A TEMP
                // trigger may bind across namespaces (lang_createtrigger.html §7); a non-temp
                // trigger's target must be in its own database (§2). The catalog then stores
                // it with the resolved target verdict.
                let (db, target) = self.trigger_target(ct)?;
                self.with_write_txn_in(db, |cat, pager| {
                    cat.create_trigger(pager, ct, source, target)
                })?;
                Ok(None)
            }
            // DROP is routed through the same catalog seam as CREATE (the catalog
            // owns `drop_object`); a rolled-back explicit DROP is resynced by the
            // catalog reload in `exec_rollback`, like any other schema change. A DROP
            // TABLE/VIEW additionally cascades any cross-namespace TEMP trigger bound to
            // the dropped object (handled inside `exec_drop`).
            Statement::Drop(drop) => {
                self.exec_drop(drop)?;
                Ok(None)
            }
            // ALTER TABLE mutates the schema (and, for DROP COLUMN, the table's row
            // data) through the same catalog seam, inside one write transaction so a
            // rejected/partial alter leaves no trace. `source` is the verbatim SQL the
            // catalog needs (e.g. ADD COLUMN recovers the new column's definition bytes
            // from it).
            //
            // KNOWN GAP (loud, honest — lang_createtrigger.html §7 cross-namespace triggers):
            // this rewrites only the TARGET store's own triggers. A cross-namespace TEMP
            // trigger bound to `(db, t)` lives in the TEMP store, so `ALTER TABLE db.t RENAME`
            // does NOT rewrite its `ON`-target / rebind it (real sqlite would). After such a
            // rename that temp trigger is left bound to the old name and stops firing until
            // re-created (a name-qualified `ForeignSchema` binding keeps resolving its db, but
            // the target TABLE name changed). The DROP-side analogue IS handled (see
            // `SqlEngine::exec_drop`'s `drop_object_with_temp_cascade` name-based cascade); the
            // rename-side cross-store rewrite is the deliberately-deferred remainder — it would
            // mirror `exec_drop` with a temp-store trigger-rewrite pass, and is out of the
            // required scope here. The overwhelmingly common same-store ALTER path is unaffected.
            Statement::AlterTable(at) => {
                let db = self.alter_target_db(&at.name)?;
                if self.namespace_live(db) {
                    self.with_write_txn_in(db, |cat, pager| cat.alter_table(pager, at, source))?;
                } else {
                    // A `temp.`-qualified ALTER before temp is materialized targets the
                    // (empty, non-live) temp store. Run `alter_table` against a throwaway
                    // empty store so the concrete catalog produces the exact live-but-empty
                    // result WITHOUT materializing temp: a `sqlite_`-reserved error
                    // (`ALTER TABLE temp.sqlite_master …` → "table sqlite_master may not be
                    // altered") or "no such table".
                    self.with_absent_namespace(|cat, pager| cat.alter_table(pager, at, source))?;
                }
                Ok(None)
            }

            // --- Maintenance statements accepted as no-ops. The property that makes a
            // no-op SAFE is that it changes ONLY PHYSICAL-STORAGE layout — never
            // query-visible content, never the schema, never an external file. Both forms
            // below preserve every row, index, rowid, and query result and create/destroy
            // no object, so they succeed exactly as real sqlite does at the SQL-content
            // level rather than aborting a script. Any variant that touches content, the
            // schema, or a file is NOT swept in — it stays a loud gap below (a no-op there
            // would fake success while diverging):
            //   * Plain `VACUUM` / `VACUUM <schema>` (`into: None`). `VACUUM ... INTO` is
            //     deliberately NOT here — it WRITES a database copy (an external file, an
            //     observable side effect), so it routes to `exec_vacuum_into` below.
            //   * `REINDEX [name]`: this engine keeps every index current on each write,
            //     so the logical index content is never stale.
            // ACCEPTED, DOCUMENTED gap (do not mistake this for byte-parity): this engine
            // DOES maintain a real free-list (minisqlite-pager `alloc`), so real sqlite's
            // VACUUM would additionally reclaim free pages (`freelist_count` → 0), shrink
            // the file (`page_count`), and defragment; REINDEX would repack index pages.
            // This no-op skips that PHYSICAL rework, so after a page-freeing workload
            // (`DELETE`/`DROP`) the `page_count`/`freelist_count` pragmas can read larger
            // here than after a real VACUUM. That physical-counter divergence is the
            // deferred remainder of real VACUUM/REINDEX (a large, separate feature), not a
            // content lie — the rows a query can read are identical.
            Statement::Vacuum { into: None, .. } => Ok(None),
            Statement::Reindex { .. } => Ok(None),

            // `VACUUM ... INTO <expr>` writes a fresh, defragmented copy of the database
            // to the file named by `<expr>`, without modifying the source (see
            // `crate::vacuum`). It is a genuine side effect (a new file with identical
            // logical content), so it is implemented rather than a no-op; the handler
            // returns no rows.
            Statement::Vacuum { into: Some(into), .. } => {
                self.exec_vacuum_into(into)?;
                Ok(None)
            }

            // --- DDL / utility / maintenance with an OBSERVABLE effect this engine does
            // not yet implement: honest, loud gaps. Faking success here would silently
            // diverge from real sqlite.
            // ANALYZE gathers table/index statistics into the internal `sqlite_stat1`
            // table (created on first use like `sqlite_sequence`). It runs inside a
            // write transaction so a rolled-back ANALYZE leaves no trace and a
            // committed one is durable. The stats are write-only: the planner does not
            // read them, so query results are unchanged (see `crate::analyze`).
            Statement::Analyze { target } => {
                self.with_write_txn(|cat, pager| crate::analyze::run(cat, pager, target.as_ref()))?;
                Ok(None)
            }
            // ATTACH / DETACH add or remove a whole database namespace on this connection
            // (see `crate::attach`). Neither returns rows.
            Statement::Attach { file, schema, key } => {
                self.exec_attach(file, schema, key)?;
                Ok(None)
            }
            Statement::Detach { schema } => {
                self.exec_detach(schema)?;
                Ok(None)
            }
            Statement::Explain(_) => Err(Error::sql("EXPLAIN not yet supported")),
            Statement::ExplainQueryPlan(_) => {
                Err(Error::sql("EXPLAIN QUERY PLAN not yet supported"))
            }

            // --- SELECT / DML: compile with the planner, run with the executor. ---
            Statement::Select(_)
            | Statement::Insert(_)
            | Statement::Update(_)
            | Statement::Delete(_) => self.run_plannable(stmt),
        }
    }

    /// Plan and execute a `SELECT`/DML statement through the planner + executor
    /// seams, then wrap its rows in a [`QueryResult`]. A mutating plan runs inside a
    /// transaction (implicit in autocommit, the open one under `BEGIN`); a read-only
    /// plan runs inside a pager READ transaction so it sees the latest committed
    /// state. See [`SqlEngine::run_mutating_collect`] / [`SqlEngine::run_reading_collect`].
    ///
    /// The planner compiles both `SELECT` and the `INSERT`/`UPDATE`/`DELETE` DML;
    /// the executor runs the resulting plan. A `mutates` plan first resets the
    /// per-statement `changes()` counter (the DML operators then advance it through
    /// the runtime); a `SELECT`/DDL/`PRAGMA` in between leaves that counter intact.
    fn run_plannable(&mut self, stmt: &Statement) -> Result<Option<QueryResult>> {
        // Plan against all live namespaces (search order: temp, main, attached). With
        // only `main` live this resolves exactly as the single-catalog planner did. The
        // borrow ends here (the owned `Plan` is returned) before the collect below.
        let mc = MultiCatalog::new(&self.catalogs, &self.namespaces);
        let plan = self.planner.plan(stmt, &mc)?;
        let rows = if plan.mutates {
            // `changes()` reports the row count of the most recent INSERT/UPDATE/
            // DELETE, so only a mutating statement resets the per-statement counter;
            // a SELECT/DDL/PRAGMA in between leaves it intact (matching sqlite's
            // sqlite3_changes()). The DML operators then advance it via the runtime.
            self.rt.reset_statement_changes();
            self.run_mutating_collect(&plan)?
        } else {
            self.run_reading_collect(&plan)?
        };
        Ok(Some(QueryResult { columns: plan.result_columns, rows }))
    }

    /// Run a mutating plan (`INSERT`/`UPDATE`/`DELETE`) and collect its rows. In
    /// autocommit it is wrapped in an implicit transaction that spans EVERY live
    /// namespace (begin-all before, commit-all after) so a mid-execution error leaves no
    /// partial write in ANY store the statement touched; inside an explicit `BEGIN` it
    /// joins the open transaction (which already spans every namespace) and neither
    /// begins nor commits.
    ///
    /// On a mid-execution error the *undo scope* depends on the DML statement's
    /// conflict-resolution algorithm (`INSERT OR …` / `UPDATE OR …`). The executor
    /// writes each pre-conflict row incrementally and then raises `Error::Constraint`
    /// at the offending row identically for ABORT/FAIL/ROLLBACK — the only difference
    /// between them is how much the engine undoes here:
    ///
    /// - **FAIL** on a constraint violation keeps the rows written before the
    ///   conflict: in autocommit, commit the implicit transaction on all live pagers to
    ///   make them durable; inside an explicit transaction, leave it open with those
    ///   rows in place. A *non-constraint* runtime error under FAIL still aborts (below).
    /// - **ROLLBACK** undoes the whole active transaction: in autocommit it degrades
    ///   to ABORT (roll back the implicit transaction on all live pagers); inside an
    ///   explicit `BEGIN` it rolls back the open transaction via
    ///   [`SqlEngine::exec_rollback`], which also clears the explicit-txn/savepoint
    ///   state and reloads the catalog caches.
    /// - **ABORT** (the default), IGNORE/REPLACE, a plain statement with no conflict
    ///   clause, and FAIL-with-a-non-constraint-error all back out just the statement
    ///   in autocommit (roll back the implicit transaction on all live pagers) and
    ///   leave an explicit transaction open.
    fn run_mutating_collect(&mut self, plan: &Plan) -> Result<Vec<Row>> {
        // An autocommit DML brackets EVERY live namespace, not just its top-level target.
        // A fired trigger / INSTEAD-OF action routes to its OWN stamped `node.db` at
        // RUNTIME (the executor writes `pagers[action.db]`), and nested-trigger recursion
        // and FK cascades likewise discover the store they write DURING execution — so a
        // static walk of the compiled plan could miss a namespace a deep chain reaches.
        // Bracketing all live pagers in one implicit transaction is robust to that: every
        // store the statement might write is staged together, so an abort rolls the WHOLE
        // statement back across all of them (the atomic unit real sqlite gives an
        // autocommit statement) and a store the statement never writes just begins/commits
        // an EMPTY overlay — a near-no-op. With only `main` live this is byte-identical to
        // the single-store begin/commit it replaces. A cross-namespace SOURCE read reads
        // the other store's committed state directly (single-connection autocommit). Inside
        // an explicit BEGIN the open transaction already spans every namespace (see
        // `exec_begin`), so this neither begins nor commits.
        let wrap = !self.txn_active();
        // Deferred-FK timing: a `DEFERRABLE INITIALLY DEFERRED` constraint (or any FK while
        // `PRAGMA defer_foreign_keys` is ON) is checked at COMMIT, not now — but ONLY when a
        // transaction is open to defer INTO. `wrap` is true exactly in autocommit (no active
        // txn), so `!wrap == txn_active()` is that predicate: in autocommit it is false and
        // every FK is enforced immediately, so deferred behaves like immediate and the
        // existing immediate-FK behavior is untouched (foreignkeys.html §4.2). The executor's
        // child/parent FK checks read this off the runtime per row.
        self.rt.set_fk_defer_active(!wrap);
        if wrap {
            self.implicit_begin_all()?;
        }
        match self.execute_plan_collect(plan) {
            Ok(rows) => {
                if wrap {
                    self.implicit_commit_all()?;
                }
                Ok(rows)
            }
            Err(e) => {
                match plan_on_conflict(plan) {
                    // OR FAIL + a constraint violation: keep the rows the statement
                    // wrote before the conflict, across every namespace it reached. In
                    // autocommit, commit the implicit txn on all live pagers (the executor
                    // already staged the pre-conflict rows); inside an explicit txn, leave
                    // it open with those rows in place. The commit's Result is ignored so a
                    // storage-level commit failure cannot mask the original constraint
                    // error `e` we return.
                    Some(OnConflict::Fail) if matches!(e, Error::Constraint(_)) => {
                        if wrap {
                            let _ = self.implicit_commit_all();
                        }
                    }
                    // OR ROLLBACK: roll back the WHOLE active transaction. In autocommit
                    // this degrades to ABORT (roll back the implicit txn on all live
                    // pagers); inside an explicit BEGIN, use the engine's own rollback (a
                    // txn is active here, so it succeeds — it also resets the
                    // in_explicit_txn/savepoint state and reloads the catalog caches). Its
                    // Result is ignored so a rollback failure cannot mask `e`.
                    Some(OnConflict::Rollback) => {
                        if wrap {
                            self.implicit_rollback_all();
                        } else {
                            let _ = self.exec_rollback();
                        }
                    }
                    // ABORT / IGNORE / REPLACE / no conflict clause / OR-FAIL with a
                    // non-constraint error: back out the statement in autocommit across
                    // every namespace it reached. An explicit txn is left open (unchanged
                    // behavior).
                    _ => {
                        if wrap {
                            self.implicit_rollback_all();
                        }
                    }
                }
                Err(e)
            }
        }
    }

    /// Run a read-only plan (`SELECT`) and collect its rows. In autocommit it is
    /// bracketed by a pager READ transaction: `begin_read` refreshes the committed
    /// snapshot to the latest (so a commit by ANOTHER connection becomes visible —
    /// the WAL snapshot advances and the rollback backing drops a stale page cache)
    /// and registers a WAL reader mark; `end_read` releases it. It is a no-op for the
    /// in-memory backing. When a transaction is already active — an explicit `BEGIN`
    /// OR one started by a `SAVEPOINT` (see [`SqlEngine::txn_active`]) — it already
    /// owns the read boundary (its `begin` refreshed the snapshot and pins it for the
    /// whole transaction — WAL snapshot isolation), so no nested read transaction is
    /// opened here.
    fn run_reading_collect(&mut self, plan: &Plan) -> Result<Vec<Row>> {
        let bracket = !self.txn_active();
        if bracket {
            self.pagers[0].begin_read()?;
        }
        let collected = self.execute_plan_collect(plan);
        if bracket {
            // Release the read snapshot whether or not the read succeeded; a release
            // failure (lock bookkeeping only) must not mask the read's own error.
            let _ = self.pagers[0].end_read();
        }
        collected
    }
}

/// The top-level DML node's conflict-resolution algorithm, if it has one. `INSERT`
/// and `UPDATE` carry an `on_conflict`; `DELETE` (and `SELECT`) do not. Used by
/// [`SqlEngine::run_mutating_collect`] to decide how much to undo on a mid-statement
/// error (the only thing that distinguishes ABORT / FAIL / ROLLBACK at the engine).
fn plan_on_conflict(plan: &Plan) -> Option<OnConflict> {
    match &plan.root {
        PlanNode::Insert(i) => Some(i.on_conflict.clone()),
        PlanNode::Update(u) => Some(u.on_conflict.clone()),
        _ => None,
    }
}
