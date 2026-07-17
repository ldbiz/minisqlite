//! `ATTACH DATABASE <file> AS <alias>` / `DETACH DATABASE <alias>` (lang_attach.html,
//! lang_detach.html): add or remove a whole database namespace on this connection.
//!
//! An attached store is one more `(Pager, SchemaCatalog, NamespaceMeta)` triple pushed onto
//! the engine's index-aligned vecs at index 2.. (main is 0, temp is 1). Everything a
//! qualified `alias.tbl` reference needs then already exists: the binder resolves the alias
//! through [`SqlEngine::db_of_schema`] (read/DML) and the DDL routers do the same, and a
//! write to `alias.tbl` brackets `pagers[alias]` — no attach-specific execution path.
//!
//! Index invariant: `temp` is FIXED at index 1, so the first ATTACH RESERVES it (via
//! `ensure_temp`) before pushing, guaranteeing attached stores never take slot 1. That
//! reservation does NOT make an unused `temp` visible to `PRAGMA database_list` — the
//! separate `temp_user_materialized` flag governs that (see [`SqlEngine`]).
//!
//! In-transaction ATTACH/DETACH (allowed since SQLite 3.21.0, and the current
//! `lang_attach.html`/`lang_detach.html` impose no prohibition):
//! * ATTACH enrolls the new pager in the open transaction (`enroll_in_open_txn`, shared
//!   with lazy temp), so a spanning COMMIT/ROLLBACK covers it.
//! * DETACH of a store that is participating in the open transaction is rejected as
//!   `locked` — a deliberate, documented narrowing (real SQLite permits detaching an
//!   attached db that is untouched within the transaction; this engine enrolls every
//!   attached store in the open transaction at ATTACH time — a pinned (DEFERRED) snapshot
//!   with its savepoint stack aligned to main — so it always participates and is never
//!   "untouched"). The enrollment pins a read snapshot rather than taking a write lock;
//!   a write to the attached store still upgrades it, and the DETACH rejection keys off
//!   `txn_active()`, not the store's lock state, so it is unchanged.

use std::path::{Path, PathBuf};

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_fileformat::DEFAULT_PAGE_SIZE;
use minisqlite_pager::{DiskPager, MemPager, Pager};
use minisqlite_sql::{Expr, Literal};
use minisqlite_types::{DbIndex, Error, NamespaceMeta, Result};

use crate::engine::SqlEngine;

/// The maximum number of simultaneously ATTACHed databases, matching SQLite's default
/// `SQLITE_LIMIT_ATTACHED` (10). SQLite counts `db->nDb >= SQLITE_LIMIT_ATTACHED + 2`
/// (main + temp + attached); the equivalent here is "attached stores (index 2..) >= 10".
const MAX_ATTACHED: usize = 10;

impl SqlEngine {
    /// `ATTACH DATABASE <file> AS <alias> [KEY <expr>]`: open the database named by `file`
    /// and register it under `alias` at the next attached index.
    ///
    /// Validation order mirrors real SQLite's `attach.c`: the attach LIMIT first, then the
    /// name-in-use check (which rejects `main`/`temp`/`temporary` and a duplicate alias with
    /// the SAME "already in use" message, since those names are all live namespaces). The
    /// store is opened only after both pass, so a rejected ATTACH never creates a file.
    ///
    /// `KEY` (encryption) is accepted syntactically and IGNORED — this build has no
    /// encryption, so there is no key to apply; a caller that supplies one gets an
    /// unencrypted database rather than an error.
    pub(crate) fn exec_attach(
        &mut self,
        file: &Expr,
        schema: &Expr,
        _key: &Option<Expr>,
    ) -> Result<()> {
        let alias = eval_attach_ident(schema, "ATTACH ... AS <name>")?;
        let filename = eval_const_filename(file)?;

        // (1) LIMIT. `attached` = live stores beyond main(0)+temp(1); temp may not be
        // reserved yet, so clamp with `saturating_sub` (attached stores force temp
        // reservation, so this is 0 exactly when there are none).
        let attached = self.namespaces.len().saturating_sub(2);
        if attached >= MAX_ATTACHED {
            return Err(Error::sql(format!(
                "too many attached databases - max {MAX_ATTACHED}"
            )));
        }
        // (2) NAME IN USE — reject if `alias` already names a live namespace: a reserved
        // built-in (main/temp/temporary) or an existing attached alias. This is precisely
        // "does the alias resolve?" — the ONE shared resolution rule (`db_of_schema` ->
        // `NamespaceMeta::resolve`), not a second copy of it — and SQLite reports every one
        // of these cases with the same "already in use" message.
        if self.db_of_schema(&alias).is_some() {
            return Err(Error::sql(format!("database {alias} is already in use")));
        }

        // (3) Open the store BEFORE mutating engine state, so a bad path fails cleanly with
        // nothing half-registered.
        let (mut pager, catalog) = open_attach_store(&filename)?;
        // (4) Reserve temp at index 1 so the attached store lands at >= 2 and temp stays
        // fixed (idempotent once temp exists). This does not mark temp user-materialized.
        self.ensure_temp()?;
        // (5) Enroll the new pager in any open transaction (like a lazy temp), so a
        // spanning COMMIT/ROLLBACK/RELEASE/ROLLBACK TO covers writes to the attached store.
        self.enroll_in_open_txn(&mut *pager)?;

        // (6) Record it — index-aligned across all three vecs. `:memory:` / a temporary
        // ("") database has no backing file, so its registry `file` is None (reported as an
        // empty `file` by `database_list`, exactly as SQLite does for an in-memory db).
        let file = backing_path(&filename);
        self.pagers.push(pager);
        self.catalogs.push(catalog);
        self.namespaces.push(NamespaceMeta::new(alias, file));
        Ok(())
    }

    /// `DETACH DATABASE <alias>`: close and remove the attached database `alias`.
    ///
    /// Rejections (spec-matched messages): `main`/`temp`/`temporary` cannot be detached
    /// ("cannot detach database <name>"); an unknown alias is "no such database: <name>";
    /// and a detach while a transaction is active is rejected as "database <name> is locked"
    /// (the store participates in the open transaction — see the module note).
    pub(crate) fn exec_detach(&mut self, schema: &Expr) -> Result<()> {
        let alias = eval_attach_ident(schema, "DETACH ... <name>")?;

        // main/temp/temporary are the fixed built-ins (index < 2) — never detachable.
        if DbIndex::from_schema_name(&alias).is_some() {
            return Err(Error::sql(format!("cannot detach database {alias}")));
        }
        // Only an attached alias remains resolvable here (built-ins handled above), so a
        // hit is always at index >= 2.
        let idx = match self.db_of_schema(&alias) {
            Some(db) => db.index(),
            None => return Err(Error::sql(format!("no such database: {alias}"))),
        };
        debug_assert!(idx >= 2, "detach resolved a built-in namespace {idx}");
        // A live transaction has enrolled every store (main committed first, then the rest);
        // the attached store is part of it, so it is locked until COMMIT/ROLLBACK.
        if self.txn_active() {
            return Err(Error::sql(format!("database {alias} is locked")));
        }

        // Remove the aligned triple. Dropping the `Box<dyn Pager>` closes its file handle.
        // Attached stores after `idx` shift down one slot, which is fine: a `DbIndex` is
        // only ever used within a single statement, and no statement is in flight here (we
        // are in autocommit — guaranteed by the txn check above). Nothing caches a `DbIndex`
        // ACROSS statements — in particular a cross-namespace TEMP trigger records its target
        // by SCHEMA NAME (`TriggerTargetDb::ForeignSchema`) or "unqualified", never a numeric
        // index, and fire-time discovery re-resolves it against this (now-renumbered) registry
        // — so a trigger bound to a higher-indexed attached db keeps firing after this shift,
        // and a later ATTACH that reuses the freed slot never inherits a stale binding.
        self.pagers.remove(idx);
        self.catalogs.remove(idx);
        self.namespaces.remove(idx);
        Ok(())
    }
}

/// Open the backing store for an ATTACH target, mirroring the engine's own constructors:
/// a fresh disk file is formatted (empty `sqlite_schema`), an existing one has its schema
/// loaded from page 1, and `:memory:` / a temporary ("") database gets a formatted
/// in-memory pager. Returns the pager + its loaded catalog, ready to push.
fn open_attach_store(filename: &str) -> Result<(Box<dyn Pager>, SchemaCatalog)> {
    if is_transient_filename(filename) {
        // `:memory:` and the empty-string "new temporary database" both map to a transient
        // in-memory backing here. A private on-disk temporary file is a documented narrow
        // gap (this build has no dedicated temp-file spill), observably identical for a
        // single connection that never exhausts memory.
        let mut pager = MemPager::new(DEFAULT_PAGE_SIZE);
        init_database(&mut pager)?;
        Ok((Box::new(pager), SchemaCatalog::new()))
    } else {
        let mut pager = DiskPager::open(Path::new(filename))?;
        if pager.page_count()? == 0 {
            init_database(&mut pager)?;
        }
        let mut catalog = SchemaCatalog::new();
        catalog.load(&pager)?;
        Ok((Box::new(pager), catalog))
    }
}

/// The registry `file` for an attached database: the raw path for a disk file (canonicalized
/// only when `PRAGMA database_list` displays it, matching `main`), or `None` for a transient
/// (`:memory:` / temporary) backing.
fn backing_path(filename: &str) -> Option<PathBuf> {
    if is_transient_filename(filename) {
        None
    } else {
        Some(PathBuf::from(filename))
    }
}

/// Whether an ATTACH filename names a transient (non-file-backed) database: the special
/// `:memory:` name or the empty string (lang_attach.html §2 — "an empty string results in a
/// new temporary database").
fn is_transient_filename(filename: &str) -> bool {
    filename.is_empty() || filename.eq_ignore_ascii_case(":memory:")
}

/// Evaluate the alias/schema-name operand, which the parser leaves as a constant expression:
/// a bare identifier (`… AS aux`) parses to an `Expr::Column`, a quoted string (`… AS 'aux'`)
/// to a text literal. Both are accepted (lang_attach.html shows the alias as a `schema-name`,
/// which may be a plain identifier or a string). A qualified/whole-column reference or any
/// non-constant expression has no attach meaning and is rejected loudly.
fn eval_attach_ident(e: &Expr, what: &str) -> Result<String> {
    match e {
        Expr::Column { schema: None, table: None, name, .. } => Ok(name.clone()),
        Expr::Literal(Literal::Text(s)) => Ok(s.clone()),
        Expr::Parenthesized(items) if items.len() == 1 => eval_attach_ident(&items[0], what),
        _ => Err(Error::sql(format!("{what} must be a database name"))),
    }
}

/// Evaluate the file operand to a constant filename string. Only a string literal (or a
/// single parenthesized one) is accepted; the facade binds no parameters here, so a
/// non-constant expression has no resolvable value and is rejected rather than guessed at.
/// (Mirrors `VACUUM INTO`'s `eval_into_filename`.)
fn eval_const_filename(e: &Expr) -> Result<String> {
    match e {
        Expr::Literal(Literal::Text(s)) => Ok(s.clone()),
        Expr::Parenthesized(items) if items.len() == 1 => eval_const_filename(&items[0]),
        _ => Err(Error::sql("ATTACH file must be a constant string filename")),
    }
}
