//! Namespace (schema) management: the lazy `temp` store and the routing that sends a
//! DDL/DML statement to the right per-namespace `(SchemaCatalog, Pager)` pair.
//!
//! A connection starts with only `main` (index 0). The `temp` store (index 1) is created
//! LAZILY on the first temp object — a fresh in-memory pager with an empty `sqlite_schema`
//! and its own schema cache — so a database that never uses temp objects keeps the exact
//! single-store shape (and byte-for-byte behavior) it had before namespaces existed.
//!
//! Routing rules (SQLite name resolution):
//! * a schema qualifier (`main`/`temp`/`temporary` or an ATTACH alias) names its namespace
//!   via [`SqlEngine::db_of_schema`]; an unknown qualifier is a loud `unknown database`
//!   error rather than a silent main write;
//! * `CREATE TEMP …` (the parsed flag) targets `temp`, and its name must be unqualified;
//! * an unqualified DROP/ALTER resolves in search order (temp, then main) to the store that
//!   actually owns the object, so `DROP TABLE t` drops the temp `t` that shadows a main one.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog, TriggerDef, TriggerTarget, TriggerTargetDb};
use minisqlite_fileformat::DEFAULT_PAGE_SIZE;
use minisqlite_pager::{MemPager, Pager};
use minisqlite_sql::{CreateIndex, CreateTrigger, Drop, DropKind, QualifiedName};
use minisqlite_types::{DbIndex, Error, NamespaceMeta, Result};

use crate::engine::SqlEngine;

impl SqlEngine {
    /// Is the `temp` namespace materialized? It is created lazily, so this is simply
    /// "are there at least two live stores" (main is always 0, temp always 1).
    pub(crate) fn temp_present(&self) -> bool {
        self.pagers.len() > DbIndex::TEMP.index()
    }

    /// Is namespace `db` LIVE — i.e. does a `(pager, catalog)` pair actually back it? A
    /// [`DbIndex`] can be resolved (by [`db_of_schema`](SqlEngine::db_of_schema)) to a slot
    /// that does not exist yet: `temp` (index 1) maps unconditionally via
    /// [`DbIndex::from_schema_name`], but its store is created LAZILY, so a `temp.` qualifier
    /// resolves to index 1 even while only `main` is live. The write-side DDL routers must
    /// check this before handing `db` to [`with_write_txn_in`](SqlEngine::with_write_txn_in),
    /// which indexes `pagers`/`catalogs` directly — a non-live index there is an
    /// out-of-bounds panic. `temp`-before-materialization is the ONLY reachable non-live
    /// case: `main` (0) is always live, and an attached alias is only in the registry while
    /// its store is live.
    pub(crate) fn namespace_live(&self, db: DbIndex) -> bool {
        db.index() < self.pagers.len()
    }

    /// Ensure namespace `db` is backed by a live `(pager, catalog)` pair before a
    /// per-database HEADER pragma WRITE indexes it. The only resolvable-but-not-live
    /// namespace is `temp` before its first object (main is always live; an attached
    /// alias is registered only while its store is live), so this MATERIALIZES `temp` on
    /// demand — matching sqlite, where `PRAGMA temp.<field> = V` persists in the
    /// (auto-created) temp database for the connection. A no-op when `db` is already
    /// live. It deliberately does NOT mark temp USER-materialized (a header write is not
    /// a user schema object), so `PRAGMA database_list` still hides an otherwise-unused
    /// temp — consistent with [`temp_user_materialized`](SqlEngine::temp_user_materialized).
    ///
    /// Any other not-live index would be an engine-invariant break; it fails closed with
    /// an error rather than pushing `temp` into the wrong slot or index-panicking.
    pub(crate) fn ensure_header_target_live(&mut self, db: DbIndex) -> Result<()> {
        if self.namespace_live(db) {
            return Ok(());
        }
        if db != DbIndex::TEMP {
            return Err(Error::sql(format!(
                "internal: header write to non-live namespace index {}",
                db.index()
            )));
        }
        self.ensure_temp()
    }

    /// The live namespaces in SQLite name-resolution SEARCH ORDER: `temp` (when present),
    /// then `main`, then any attached (2..). Mirrors `MultiCatalog`'s own order, so the
    /// engine's routing and the planner's resolution agree on which store owns a name.
    pub(crate) fn live_search_order(&self) -> impl Iterator<Item = DbIndex> + '_ {
        let temp = self.temp_present().then_some(DbIndex::TEMP);
        temp.into_iter()
            .chain(std::iter::once(DbIndex::MAIN))
            .chain((2..self.pagers.len()).map(|i| DbIndex(i as u16)))
    }

    /// Create the `temp` namespace if it does not exist yet: a fresh in-memory pager
    /// formatted with an empty `sqlite_schema` (page 1) plus an empty schema cache, pushed
    /// at index [`DbIndex::TEMP`]. Idempotent — a second call is a no-op — and NEVER touches
    /// the main file, so temp objects are transient by construction.
    pub(crate) fn ensure_temp(&mut self) -> Result<()> {
        if self.temp_present() {
            return Ok(());
        }
        // The invariant the rest of the engine relies on: temp is exactly index 1, so it can
        // only be created when main (index 0) is the sole store. Assert it rather than
        // silently pushing temp to the wrong slot.
        debug_assert_eq!(self.pagers.len(), DbIndex::TEMP.index());
        let mut pager = MemPager::new(DEFAULT_PAGE_SIZE);
        // `init_database` auto-commits page 1 (an empty `sqlite_schema`) as temp's baseline
        // — the state a ROLLBACK reverts to (the empty temp db survives, its objects do not).
        init_database(&mut pager)?;
        // A temp store created mid-transaction JOINS the open transaction at the current
        // depth (shared with ATTACH — see `enroll_in_open_txn`), so a spanning COMMIT /
        // ROLLBACK / RELEASE / ROLLBACK TO covers it identically to main.
        self.enroll_in_open_txn(&mut pager)?;
        self.pagers.push(Box::new(pager));
        self.catalogs.push(SchemaCatalog::new());
        self.namespaces.push(NamespaceMeta::new("temp", None));
        Ok(())
    }

    /// Run a DDL catalog mutation against a THROWAWAY empty namespace, discarding it after —
    /// the emulation for a schema qualifier that resolves to a NON-LIVE store. The only
    /// reachable such case is `temp` before its first object (see
    /// [`namespace_live`](SqlEngine::namespace_live)); the write-DDL dispatch arms
    /// ([`crate::dispatch`]) route it here instead of
    /// [`with_write_txn_in`](SqlEngine::with_write_txn_in).
    ///
    /// A `temp.`-qualified DROP/ALTER/CREATE INDEX issued before temp
    /// exists must behave EXACTLY as it would against a live-but-EMPTY temp store, WITHOUT
    /// materializing temp. Rather than re-spell each concrete-catalog message in the router
    /// (the `sqlite_`-reserved pre-checks that fire even under `IF EXISTS`, the `IF EXISTS`
    /// no-op, and "no such …") — a second source of truth that would drift from
    /// [`SchemaCatalog`] — this runs the REAL mutation against a fresh empty store built
    /// exactly like [`ensure_temp`](SqlEngine::ensure_temp) does (a `MemPager` with an empty
    /// `sqlite_schema` plus an empty [`SchemaCatalog`]), so the catalog emits every message
    /// verbatim, then drops the scratch store. Because it is never pushed onto `self` and
    /// `temp_user_materialized` is untouched, temp stays neither reserved nor
    /// user-materialized. No transaction bracketing is needed: the scratch store is discarded
    /// WHOLESALE, so even a hypothetical partial page write cannot escape (worst case is a
    /// wrong `Ok`, never a panic or persisted corruption). In practice only the
    /// validate-before-write DDLs are routed here (`create_index`/`drop_object`/`alter_table`
    /// all reject a reserved/missing name before their first page write), so an empty store
    /// errors/no-ops before writing at all — do NOT route a DDL that could succeed-with-write
    /// on an empty store through here. `&self` (the scratch store is self-contained) so it
    /// composes with the routers' `&self` resolution without a conflicting borrow.
    pub(crate) fn with_absent_namespace<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut SchemaCatalog, &mut dyn Pager) -> Result<()>,
    {
        let mut pager = MemPager::new(DEFAULT_PAGE_SIZE);
        init_database(&mut pager)?;
        let mut catalog = SchemaCatalog::new();
        f(&mut catalog, &mut pager)
    }

    /// Bring a freshly-opened store's pager into the connection's CURRENT transaction, so a
    /// namespace created mid-transaction (a lazy `temp`, an `ATTACH`ed file) is covered by
    /// the same COMMIT/ROLLBACK/RELEASE/ROLLBACK TO as every other live store. The engine
    /// keeps every pager's savepoint stack in lockstep (contiguous depths 0..k-1), so the
    /// new pager must catch up to the current depth:
    ///  * BEGIN-started (`in_explicit_txn`): open its transaction DEFERRED
    ///    ([`Pager::begin_deferred`]) so it pins a read snapshot without eagerly taking
    ///    its own write lock — the first write to the new store upgrades on demand, and an
    ///    outermost RELEASE keeps it open (the enclosing BEGIN holds it) — like main.
    ///  * SAVEPOINT-started in autocommit: do NOT begin first — the first replayed
    ///    `savepoint()` implicitly begins it (also DEFERRED) AND marks it
    ///    started-by-savepoint, so an outermost RELEASE commits it — again matching main's
    ///    outermost savepoint.
    /// Replaying `savepoints.len()` savepoints then yields depths that align with main's.
    /// In plain autocommit (no active txn) this is a no-op and the new store stays in
    /// autocommit too. `pager` is the not-yet-pushed store, borrowed disjointly from the
    /// engine's txn state (`&self`).
    ///
    /// DEFERRED here mirrors the rule the rest of the engine follows (`exec_begin`'s
    /// default `BEGIN`, `Cow::savepoint`'s implicit begin): a transaction start that may
    /// stay read-only must not eagerly grab the WAL write lock. For the common enrolled
    /// store — the lazy in-memory `temp` pager — `begin_deferred == begin` observably, so
    /// this is a no-op there; it matters only for an `ATTACH`ed WAL file enrolled inside
    /// an open transaction (a deep edge), where deferring correctly leaves a concurrent
    /// writer of that file unblocked until the attached store is actually written. (An
    /// enclosing `BEGIN IMMEDIATE` still eager-locks `main`; an attached store written
    /// under it upgrades on its first write rather than at ATTACH time — an accepted,
    /// out-of-scope nuance for cross-connection ATTACH.)
    pub(crate) fn enroll_in_open_txn(&self, pager: &mut dyn Pager) -> Result<()> {
        if self.txn_active() {
            if self.in_explicit_txn {
                pager.begin_deferred()?;
            }
            for _ in 0..self.savepoints.len() {
                pager.savepoint()?;
            }
        }
        Ok(())
    }

    /// The namespace a `CREATE TABLE`/`VIEW`/`TRIGGER` targets, creating the `temp` store
    /// first when needed. `temp_flag` is the parsed `TEMP`/`TEMPORARY` keyword.
    ///
    /// * `CREATE TEMP …` → `temp`, and the name must be UNQUALIFIED (SQLite: "temporary
    ///   table name must be unqualified"); a qualifier alongside the flag is that error.
    /// * `CREATE TABLE temp.x` (a `temp`/`temporary` qualifier, no flag) → `temp`.
    /// * `CREATE TABLE main.x` or an unqualified non-temp create → `main`.
    /// * `CREATE TABLE aux.x` (an ATTACH alias) → the attached store.
    /// * any other qualifier → `unknown database`.
    ///
    /// Materializing `temp` here (either TEMP path) marks it USER-materialized so
    /// `PRAGMA database_list` lists it (see [`SqlEngine::temp_user_materialized`]).
    pub(crate) fn create_target_db(
        &mut self,
        temp_flag: bool,
        name: &QualifiedName,
    ) -> Result<DbIndex> {
        if temp_flag {
            if name.schema.is_some() {
                return Err(Error::sql("temporary table name must be unqualified"));
            }
            self.ensure_temp()?;
            self.temp_user_materialized = true;
            return Ok(DbIndex::TEMP);
        }
        match name.schema.as_deref() {
            None => Ok(DbIndex::MAIN),
            // Resolve `main`/`temp`/an ATTACH alias through the one registry rule.
            Some(schema) => match self.db_of_schema(schema) {
                Some(DbIndex::TEMP) => {
                    self.ensure_temp()?;
                    self.temp_user_materialized = true;
                    Ok(DbIndex::TEMP)
                }
                Some(db) => Ok(db),
                None => Err(Error::sql(format!("unknown database {schema}"))),
            },
        }
    }

    /// Resolve a `CREATE TRIGGER`: WHERE the trigger lives, and the [`TriggerTarget`] verdict
    /// its store needs (`lang_createtrigger.html`).
    ///
    /// The trigger lives where [`create_target_db`](SqlEngine::create_target_db) puts any
    /// `CREATE [TEMP] …` object — the trigger NAME's schema qualifier, or `temp` for a `TEMP`
    /// trigger (materializing the temp store), else `main`. Given that `trigger_db`:
    ///
    /// * NON-TEMP trigger — the target must live in the trigger's OWN database (§2, enforced by
    ///   SQLite's schema fixer). A target EXPLICITLY qualified to a DIFFERENT database is the
    ///   rejected form (`CREATE TRIGGER main.trg ON aux.t`, or the unqualified-name-defaults-to-
    ///   main `CREATE TRIGGER trg ON aux.t`). Otherwise the target is resolved in the trigger's
    ///   own store as [`TriggerTarget::SameStore`] — byte-for-byte the pre-namespace path, so the
    ///   overwhelmingly common trigger case (and its error order) is unchanged.
    /// * TEMP trigger — exempt from the same-database rule (§7 "TEMP Triggers on Non-TEMP
    ///   Tables"): the target is resolved across namespaces (qualified → that store; unqualified
    ///   → search order temp, main, attached). Found in `temp` (== `trigger_db`) or an
    ///   unqualified target found NOWHERE → `SameStore` (the temp store validates / emits "no
    ///   such table" in order); found in another store → [`TriggerTarget::Foreign`] carrying
    ///   the ON-clause qualifier (or `None` = unqualified) + kind, so the store binds by NAME.
    ///
    /// A QUALIFIED `ON schema.name` is HONORED, never silently rebound: an unknown qualifier
    /// errors `unknown database` and a live-but-absent qualified target errors
    /// `no such table: schema.name` — for a TEMP trigger too (real SQLite's `sqlite3SrcListLookup`
    /// on an explicit `schema.tbl`), rather than falling through to the temp store.
    pub(crate) fn trigger_target(
        &mut self,
        ct: &CreateTrigger,
    ) -> Result<(DbIndex, TriggerTarget)> {
        // Where the trigger LIVES (materializes temp for a TEMP trigger, like a temp table; also
        // enforces "temporary trigger name must be unqualified" for `CREATE TEMP … name`).
        let trigger_db = self.create_target_db(ct.temp, &ct.name)?;
        let on = &ct.table;

        match on.schema.as_deref() {
            // ---- QUALIFIED `ON schema.name`: the qualifier is authoritative ----------------
            Some(schema) => {
                // An unknown qualifier is a loud `unknown database` (never a silent bind to the
                // trigger's own store by bare name), mirroring every other DDL router.
                let target_db = self
                    .db_of_schema(schema)
                    .ok_or_else(|| Error::sql(format!("unknown database {schema}")))?;
                if target_db == trigger_db {
                    // SAME database as the trigger (a same-db qualifier, or the trigger's own
                    // db): the store validates existence + kind against its OWN cache, so the
                    // message ("no such table: name") and error ORDERING are byte-for-byte the
                    // pre-namespace same-store path — unchanged for the common qualified case.
                    return Ok((trigger_db, TriggerTarget::SameStore));
                }
                // DIFFERENT database than the trigger.
                if !ct.temp {
                    // §2 (SQLite's schema fixer): a NON-TEMP trigger may not target another db.
                    return Err(Error::sql(format!(
                        "trigger {} cannot reference objects in database {}",
                        ct.name.name, schema
                    )));
                }
                // TEMP trigger, FOREIGN target (§7): the qualifier is HONORED, never silently
                // discarded/rebound to the temp store — the target must EXIST in the named
                // store, else a qualifier-preserving "no such table: schema.name" (a non-live
                // qualified store, e.g. temp before it exists, holds nothing → equally absent).
                let kind = if self.namespace_live(target_db) {
                    let cat = &self.catalogs[target_db.index()];
                    if cat.table(&on.name)?.is_some() {
                        Some(false)
                    } else if cat.view(&on.name)?.is_some() {
                        Some(true)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let Some(is_view) = kind else {
                    return Err(Error::sql(format!("no such table: {schema}.{}", on.name)));
                };
                // Bind by the schema NAME so fire time re-resolves it across DETACH remaps.
                Ok((trigger_db, TriggerTarget::Foreign { schema: Some(schema.to_string()), is_view }))
            }
            // ---- UNQUALIFIED `ON name` -----------------------------------------------------
            None => {
                if !ct.temp {
                    // Non-TEMP: the target must be in the trigger's own store; resolve there
                    // (SameStore — byte-for-byte the pre-namespace path, incl. its "no such
                    // table"). A non-TEMP trigger never searches another namespace.
                    return Ok((trigger_db, TriggerTarget::SameStore));
                }
                // TEMP: resolve across namespaces by search order (temp, main, attached).
                let mut resolved: Option<(DbIndex, bool)> = None;
                for db in self.live_search_order() {
                    let cat = &self.catalogs[db.index()];
                    if cat.table(&on.name)?.is_some() {
                        resolved = Some((db, false));
                        break;
                    }
                    if cat.view(&on.name)?.is_some() {
                        resolved = Some((db, true));
                        break;
                    }
                }
                match resolved {
                    // Found nowhere → defer to the temp store's own "no such table" (SameStore),
                    // preserving the reserved-name / duplicate / no-such-table ordering.
                    None => Ok((trigger_db, TriggerTarget::SameStore)),
                    Some((db, _)) if db == trigger_db => Ok((trigger_db, TriggerTarget::SameStore)),
                    // Cross-namespace, unqualified: bound as ForeignUnqualified (re-resolved by
                    // search order at fire time — the doc's reattach-on-schema-change behavior).
                    Some((_, is_view)) => {
                        Ok((trigger_db, TriggerTarget::Foreign { schema: None, is_view }))
                    }
                }
            }
        }
    }

    /// The namespace a `CREATE INDEX` targets — the store that owns its TARGET TABLE (an
    /// index always lives beside its table). `CREATE INDEX` has no `TEMP` keyword, so the
    /// namespace comes from the index name's schema qualifier when present, else from the
    /// table's own namespace resolved in search order. A missing table falls back to `main`
    /// so `create_index` there reports SQLite's canonical "no such table".
    ///
    /// A `temp.` qualifier resolves to [`DbIndex::TEMP`] even before the lazy temp store
    /// exists (a NON-LIVE index): this only RESOLVES the target, so the `CreateIndex`
    /// dispatch arm detects that with [`namespace_live`](SqlEngine::namespace_live) and runs
    /// the statement against a throwaway empty store
    /// ([`with_absent_namespace`](SqlEngine::with_absent_namespace)) rather than materializing
    /// temp here.
    pub(crate) fn index_target_db(&self, ci: &CreateIndex) -> Result<DbIndex> {
        if let Some(schema) = ci.name.schema.as_deref() {
            return self
                .db_of_schema(schema)
                .ok_or_else(|| Error::sql(format!("unknown database {schema}")));
        }
        for db in self.live_search_order() {
            if self.catalogs[db.index()].table(&ci.table)?.is_some() {
                return Ok(db);
            }
        }
        Ok(DbIndex::MAIN)
    }

    /// The namespace a `DROP {TABLE|INDEX|VIEW|TRIGGER}` targets: the explicit qualifier, or
    /// — for an unqualified name — the first store in search order that actually defines an
    /// object of THAT kind by that name (so `DROP TABLE t` drops the temp `t` shadowing a
    /// main one, while `DROP INDEX x` skips a temp TABLE `x` and finds the main index). A
    /// name found nowhere falls back to `main`, where `drop_object` yields the canonical
    /// "no such {kind}" (or the `IF EXISTS` no-op).
    pub(crate) fn drop_target_db(&self, drop: &Drop) -> Result<DbIndex> {
        if let Some(schema) = drop.name.schema.as_deref() {
            return self
                .db_of_schema(schema)
                .ok_or_else(|| Error::sql(format!("unknown database {schema}")));
        }
        let name = &drop.name.name;
        for db in self.live_search_order() {
            let cat = &self.catalogs[db.index()];
            let defines = match drop.kind {
                DropKind::Table => cat.table(name)?.is_some(),
                DropKind::Index => cat.index(name)?.is_some(),
                DropKind::View => cat.view(name)?.is_some(),
                DropKind::Trigger => cat.has_trigger_named(name),
            };
            if defines {
                return Ok(db);
            }
        }
        Ok(DbIndex::MAIN)
    }

    /// Execute a `DROP {TABLE|INDEX|VIEW|TRIGGER}`. Routes the object's own removal to the
    /// store that owns it ([`drop_target_db`](SqlEngine::drop_target_db)), then — for a
    /// `DROP TABLE`/`DROP VIEW` of a NON-temp object — cascade-drops any cross-namespace TEMP
    /// trigger bound to it (`lang_createtrigger.html` §7), which lives in the temp store the
    /// target store cannot reach.
    ///
    /// The common case (no temp store, or no cross-namespace trigger, or a `DROP INDEX`/`DROP
    /// TRIGGER`) takes the unchanged single-store path. When a cascade IS needed the target
    /// drop and the temp-trigger removal run as ONE atomic unit: in autocommit both are
    /// bracketed in an implicit transaction across every live namespace (like a
    /// cross-namespace DML), so a failure backs BOTH out; inside an open transaction they just
    /// join it and a later COMMIT/ROLLBACK covers both.
    pub(crate) fn exec_drop(&mut self, drop: &Drop) -> Result<()> {
        // Resolve which namespace owns the object (temp shadows main for an unqualified name;
        // a qualifier pins the store directly).
        let db = self.drop_target_db(drop)?;

        // Does this drop need the cross-namespace temp-trigger cascade? Only a DROP TABLE/VIEW
        // of an object in a NON-temp store, and only when the temp store actually holds a
        // cross-namespace trigger. Otherwise the fast single-store path below is unchanged.
        let needs_temp_cascade = matches!(drop.kind, DropKind::Table | DropKind::View)
            && db != DbIndex::TEMP
            && self.temp_present()
            && self.catalogs[DbIndex::TEMP.index()].has_cross_namespace_triggers();

        if !needs_temp_cascade {
            if self.namespace_live(db) {
                self.with_write_txn_in(db, |cat, pager| cat.drop_object(pager, drop))?;
            } else {
                // A `temp.`-qualified DROP before temp is materialized targets the (empty,
                // non-live) temp store. Run `drop_object` against a throwaway empty store so
                // the concrete catalog produces the exact live-but-empty result WITHOUT
                // materializing temp: a `sqlite_`-reserved error that fires even under IF
                // EXISTS (`DROP TABLE [IF EXISTS] temp.sqlite_master` → "table sqlite_master
                // may not be dropped", NOT a silent success), an `IF EXISTS` no-op for an
                // ordinary absent object, or "no such <kind>".
                self.with_absent_namespace(|cat, pager| cat.drop_object(pager, drop))?;
            }
            return Ok(());
        }

        // Atomic two-store drop (target object + cross-namespace temp triggers bound to it).
        let wrap = !self.txn_active();
        if wrap {
            self.implicit_begin_all()?;
        }
        match self.drop_object_with_temp_cascade(db, drop) {
            Ok(()) => {
                if wrap {
                    self.implicit_commit_all()?;
                }
                Ok(())
            }
            Err(e) => {
                if wrap {
                    // Back both stores out, then resync their caches to the rolled-back page 1
                    // (the cache-last drops may have mutated a cache before the failure). Only
                    // in autocommit: inside an open txn a failed DDL leaves the cache as-is and
                    // the eventual ROLLBACK reloads it, matching `with_write_txn_in`.
                    self.implicit_rollback_all();
                    let _ = self.catalogs[db.index()].load(&*self.pagers[db.index()]);
                    let ti = DbIndex::TEMP.index();
                    let _ = self.catalogs[ti].load(&*self.pagers[ti]);
                }
                Err(e)
            }
        }
    }

    /// Drop the target object in `db` and then every cross-namespace temp trigger bound to
    /// `(db, name)` from the temp store — the two mutations of a cross-namespace `DROP
    /// TABLE`/`DROP VIEW`. `catalogs`/`pagers` are distinct fields, so each store's `(catalog,
    /// pager)` is a disjoint borrow. The caller wraps this in the transaction bracket.
    ///
    /// The temp-store bindings are NAMES, not indices, so the ENGINE (which holds the registry
    /// and search order) resolves which foreign triggers bind to `(db, name)` and passes their
    /// names to the temp store — the store cannot resolve a schema name or search order itself.
    /// Victims are computed BEFORE the object drop so an unqualified (search-order) binding
    /// still sees the object present.
    fn drop_object_with_temp_cascade(&mut self, db: DbIndex, drop: &Drop) -> Result<()> {
        let victims = self.temp_foreign_trigger_names_bound_to(db, &drop.name.name)?;
        {
            let i = db.index();
            let cat = &mut self.catalogs[i];
            let pager = &mut *self.pagers[i];
            cat.drop_object(pager, drop)?;
        }
        if !victims.is_empty() {
            let ti = DbIndex::TEMP.index();
            let cat = &mut self.catalogs[ti];
            let pager = &mut *self.pagers[ti];
            cat.drop_triggers_by_folded_name(pager, &victims)?;
        }
        Ok(())
    }

    /// The folded names of the cross-namespace TEMP triggers currently bound to `(db, name)`
    /// — the victims of a `DROP TABLE`/`DROP VIEW` on that `(db, object)`. Resolves each temp
    /// foreign trigger's binding against the LIVE registry/search order (the same rule as
    /// fire-time discovery), so a `DETACH`-remapped or unqualified binding is matched
    /// correctly. Empty when there is no temp store.
    fn temp_foreign_trigger_names_bound_to(
        &self,
        db: DbIndex,
        name: &str,
    ) -> Result<Vec<String>> {
        let ti = DbIndex::TEMP.index();
        if ti >= self.catalogs.len() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for t in self.catalogs[ti].triggers_on(name)? {
            if self.foreign_binds_to(t, db)? {
                out.push(t.name.to_ascii_lowercase());
            }
        }
        Ok(out)
    }

    /// Does the cross-namespace temp trigger `t` bind to namespace `db` RIGHT NOW? Mirrors
    /// [`MultiCatalog`](minisqlite_catalog::MultiCatalog)'s fire-time rule so create-time DROP
    /// resolution and fire-time discovery agree: a
    /// [`ForeignSchema`](TriggerTargetDb::ForeignSchema) name resolves through the live
    /// registry; a [`ForeignUnqualified`](TriggerTargetDb::ForeignUnqualified) target resolves
    /// by search order. A same-store def never matches (callers pass temp-store triggers only).
    fn foreign_binds_to(&self, t: &TriggerDef, db: DbIndex) -> Result<bool> {
        match &t.target {
            TriggerTargetDb::SameStore => Ok(false),
            TriggerTargetDb::ForeignSchema(schema) => Ok(self.db_of_schema(schema) == Some(db)),
            TriggerTargetDb::ForeignUnqualified => Ok(self.name_owner_db(&t.table)? == Some(db)),
        }
    }

    /// The first namespace in search order (temp, main, attached) that defines `name` as a
    /// table or view — the engine mirror of `MultiCatalog::owner_of`, used to resolve an
    /// unqualified foreign trigger binding for the DROP cascade.
    fn name_owner_db(&self, name: &str) -> Result<Option<DbIndex>> {
        for db in self.live_search_order() {
            let cat = &self.catalogs[db.index()];
            if cat.table(name)?.is_some() || cat.view(name)?.is_some() {
                return Ok(Some(db));
            }
        }
        Ok(None)
    }

    /// The namespace an `ALTER TABLE` targets: the explicit qualifier, or the first store in
    /// search order that owns the named TABLE. A table found nowhere falls back to `main`,
    /// where `alter_table` reports the canonical "no such table".
    ///
    /// As with [`index_target_db`](SqlEngine::index_target_db), a `temp.` qualifier can
    /// resolve to a NON-LIVE temp namespace; the `AlterTable` dispatch arm routes that to the
    /// throwaway empty-store emulation
    /// ([`with_absent_namespace`](SqlEngine::with_absent_namespace)), so this never
    /// materializes temp.
    pub(crate) fn alter_target_db(&self, name: &QualifiedName) -> Result<DbIndex> {
        if let Some(schema) = name.schema.as_deref() {
            return self
                .db_of_schema(schema)
                .ok_or_else(|| Error::sql(format!("unknown database {schema}")));
        }
        for db in self.live_search_order() {
            if self.catalogs[db.index()].table(&name.name)?.is_some() {
                return Ok(db);
            }
        }
        Ok(DbIndex::MAIN)
    }
}
