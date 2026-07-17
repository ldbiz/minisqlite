//! The `Catalog` seam trait: the one interface every schema store presents. It is
//! a fixed engine seam — declared exactly once in this crate (enforced by
//! `minisqlite/tests/seams.rs`), so the route to the schema cannot fork. The
//! concrete store ([`SchemaCatalog`](crate::SchemaCatalog)) implements it over the
//! page-1 `sqlite_schema` b-tree with an in-memory cache.

use crate::def::{IndexDef, TableDef};
use minisqlite_pager::Pager;
use minisqlite_sql::{AlterTable, CreateIndex, CreateTable, CreateTrigger, CreateView, Drop};
use minisqlite_types::{DbIndex, Result};

/// The single schema store. Lookups borrow (`&TableDef` / `&IndexDef`) so they do
/// not copy, and the same store backs both definition and query, so there is
/// exactly one source of truth for what exists.
///
/// The schema is persisted in the `sqlite_schema` b-tree at page 1 (the on-disk
/// source of truth); the store keeps an in-memory case-insensitive cache over it so
/// lookups are O(1). `create_*` write the `sqlite_schema` row and update the cache;
/// [`load`](Catalog::load) rebuilds the cache by scanning page 1. The same path
/// serves an in-memory database (a valid SQLite image with a real page-1 schema) and
/// an on-disk file.
///
/// SQLite identifiers are case-insensitive over ASCII, so every implementation
/// matches names case-insensitively: a table created as `t` is found by `T`. The
/// original spelling is preserved inside the returned def for reporting.
///
/// The `create_*` methods run within whatever transaction the caller has open; they
/// do not begin/commit/rollback the pager themselves.
pub trait Catalog {
    /// Borrow a table's schema by name, or `None` if it does not exist.
    fn table(&self, name: &str) -> Result<Option<&TableDef>>;

    /// Borrow an index's schema by name, or `None` if it does not exist.
    fn index(&self, name: &str) -> Result<Option<&IndexDef>>;

    /// Borrow a view's definition by name, or `None` if it does not exist. A view has
    /// no b-tree, so it is NOT a table — [`table`](Catalog::table) returns `None` for a
    /// view name — and this is the separate read seam the planner consults to EXPAND a
    /// view referenced in a query's FROM: it re-parses the returned [`ViewDef`]'s stored
    /// `CREATE VIEW … AS <select>` text and inlines that SELECT like a derived table.
    /// Names match case-insensitively, like every other lookup.
    ///
    /// A default body returning `Ok(None)` is provided (mirroring the `create_*` write
    /// methods' default bodies), so a read-only or test [`Catalog`] that models no views
    /// need not implement it; only the real store
    /// ([`SchemaCatalog`](crate::SchemaCatalog)) overrides it.
    fn view(&self, name: &str) -> Result<Option<&crate::def::ViewDef>> {
        let _ = name;
        Ok(None)
    }

    /// Borrow the indexes defined on a table, so the planner can choose an index
    /// access path over a full scan. Returned in creation order. An empty vector
    /// means the table has no indexes (or does not exist); this is not an error,
    /// so the planner can ask without first checking the table exists.
    fn indexes_on<'a>(&'a self, table: &str) -> Result<Vec<&'a IndexDef>>;

    /// Borrow the triggers defined ON a table (matched against each trigger's TARGET
    /// table, [`TriggerDef::table`](crate::def::TriggerDef::table)), so the trigger
    /// compiler/executor can find every trigger a write to `table` must fire. Names
    /// match case-insensitively, like every other lookup. An empty vector means the
    /// table has no triggers (or does not exist); this is not an error, so a caller
    /// can ask without first checking the table exists.
    ///
    /// The order is DETERMINISTIC (see the concrete store for the ordering rule),
    /// because a table's same-event triggers fire in a fixed sequence and the plan
    /// each write compiles must be reproducible run to run.
    ///
    /// A default body returning an empty vector is provided (mirroring
    /// [`view`](Catalog::view)), so a read-only or test [`Catalog`] that models no
    /// triggers need not implement it; only the real store
    /// ([`SchemaCatalog`](crate::SchemaCatalog)) overrides it.
    fn triggers_on(&self, table: &str) -> Result<Vec<&crate::def::TriggerDef>> {
        let _ = table;
        Ok(Vec::new())
    }

    /// Borrow every user TABLE definition currently in the schema (NOT views, NOT
    /// indexes; excludes the internal `sqlite_*` schema tables — `sqlite_schema`,
    /// `sqlite_sequence`, and the like). The order is unspecified. This is the
    /// enumeration seam FOREIGN KEY enforcement needs: to apply a parent-side
    /// referential action (`ON DELETE`/`ON UPDATE` of a parent row) it must find the
    /// CHILD tables whose `foreign_keys[*].parent_table` names that parent, and there
    /// is otherwise no way to walk every table (the other lookups are all by name).
    ///
    /// A default body returning an empty vector is provided (mirroring
    /// [`view`](Catalog::view) and [`triggers_on`](Catalog::triggers_on)), so a
    /// read-only or test [`Catalog`] need not implement it; only the real store
    /// ([`SchemaCatalog`](crate::SchemaCatalog)) returns real data, and FK enforcement
    /// runs only through it.
    fn tables(&self) -> Result<Vec<&TableDef>> {
        Ok(Vec::new())
    }

    // -----------------------------------------------------------------------
    // Namespace-qualified read lookups.
    //
    // A single connection can hold several schema stores (`main`, `temp`, attached),
    // each addressed by a [`DbIndex`]. The bare read methods above are *namespace-blind*:
    // on a single store they hit that store; on a composite ([`MultiCatalog`]) they run
    // SQLite's name-resolution SEARCH ORDER (temp, then main, then attached — first match
    // wins). The `*_in` methods below instead target ONE explicit namespace, which is what
    // the binder-resolved `db` on a plan node names.
    //
    // The distinction is load-bearing under SHADOWING: with a temp table `t` shadowing a
    // main `t`, the binder resolves `main.t` to `DbIndex::MAIN` and stamps that on the
    // plan node; the executor must then open the cursor in `main`, so it looks the def up
    // via `table_in(MAIN, "t")` — NOT bare `table("t")`, which the search order would
    // resolve to the temp one. A def and the pager the cursor opens on must agree on the
    // namespace, or the root page is read from the wrong store.
    //
    // Every default body IGNORES `db` and delegates to the bare method — correct for any
    // single-namespace store (there is only `main`, so a `db` can only be `MAIN`). Only
    // the composite [`MultiCatalog`] overrides them to route to `catalogs[db]`.

    /// Borrow a table's schema FROM the `db` namespace (see the block comment above), or
    /// `None`. Default: ignore `db`, delegate to [`table`](Catalog::table).
    fn table_in(&self, db: DbIndex, name: &str) -> Result<Option<&TableDef>> {
        let _ = db;
        self.table(name)
    }

    /// Borrow an index's schema FROM the `db` namespace, or `None`. Default: ignore `db`,
    /// delegate to [`index`](Catalog::index).
    fn index_in(&self, db: DbIndex, name: &str) -> Result<Option<&IndexDef>> {
        let _ = db;
        self.index(name)
    }

    /// Borrow a view's definition FROM the `db` namespace, or `None`. Default: ignore
    /// `db`, delegate to [`view`](Catalog::view).
    fn view_in(&self, db: DbIndex, name: &str) -> Result<Option<&crate::def::ViewDef>> {
        let _ = db;
        self.view(name)
    }

    /// Borrow the indexes defined on a table IN the `db` namespace (an index always lives
    /// in the same namespace as its table). Default: ignore `db`, delegate to
    /// [`indexes_on`](Catalog::indexes_on).
    fn indexes_on_in<'a>(&'a self, db: DbIndex, table: &str) -> Result<Vec<&'a IndexDef>> {
        let _ = db;
        self.indexes_on(table)
    }

    /// Borrow the triggers whose TARGET table is `table`, defined IN the `db` namespace.
    /// Default: ignore `db`, delegate to [`triggers_on`](Catalog::triggers_on).
    fn triggers_on_in(&self, db: DbIndex, table: &str) -> Result<Vec<&crate::def::TriggerDef>> {
        let _ = db;
        self.triggers_on(table)
    }

    /// Borrow every user table IN the `db` namespace — the per-namespace form of
    /// [`tables`](Catalog::tables). FOREIGN KEY enforcement uses this: SQLite forbids a
    /// cross-database FK, so a write to a table in `db` only needs the child tables in the
    /// SAME `db`. Default: ignore `db`, delegate to [`tables`](Catalog::tables).
    fn tables_in(&self, db: DbIndex) -> Result<Vec<&TableDef>> {
        let _ = db;
        self.tables()
    }

    /// The namespace an UNQUALIFIED object reference resolves to — the first store, in
    /// SQLite search order (temp, then main, then attached), that defines `name` as a
    /// TABLE or a VIEW — or `None` if no live namespace defines it. This is the one fact
    /// the binder cannot derive from the `*_in` lookups alone: for a bare name it must
    /// learn WHICH namespace won the search so it can stamp that [`DbIndex`] on the plan
    /// node (a temp table shadows a same-named main one, so the answer is `TEMP` there).
    ///
    /// It considers tables AND views because a FROM/DML name resolves to either kind in
    /// the same search order; the caller then fetches the def with `table_in`/`view_in`
    /// at the returned `db` and decides what to do per kind.
    ///
    /// Default: the single store owns every name it defines, so this is `Some(MAIN)` when
    /// `name` is a table or view here, else `None`. Only the composite
    /// [`MultiCatalog`](crate::MultiCatalog) overrides it to run the real search order.
    fn owner_db(&self, name: &str) -> Result<Option<DbIndex>> {
        if self.table(name)?.is_some() || self.view(name)?.is_some() {
            Ok(Some(DbIndex::MAIN))
        } else {
            Ok(None)
        }
    }

    /// The namespace an unqualified INDEX name resolves to — the first store, in SQLite
    /// search order, that defines `name` as an INDEX — or `None` if none does. This is the
    /// index counterpart of [`owner_db`](Catalog::owner_db) (which resolves a table/view
    /// name): an index name is NOT a table/view name, so the two cannot share one lookup.
    /// The introspection pragmas whose argument is an INDEX (`index_info`, `index_xinfo`)
    /// resolve their unqualified target through here, then fetch the def with `index_in` and
    /// the owning table with `table_in` at the returned `db`.
    ///
    /// Default: the single store owns every index it defines, so this is `Some(MAIN)` when
    /// `name` is an index here, else `None`. Only the composite
    /// [`MultiCatalog`](crate::MultiCatalog) overrides it to run the real search order.
    fn owner_db_of_index(&self, name: &str) -> Result<Option<DbIndex>> {
        if self.index(name)?.is_some() {
            Ok(Some(DbIndex::MAIN))
        } else {
            Ok(None)
        }
    }

    /// The namespace a schema QUALIFIER (`main.t`, `temp.t`, `aux.t`) names, or `None` for
    /// an unknown/unattached qualifier (which the caller reports as "no such table"). This
    /// is the qualified counterpart to [`owner_db`](Catalog::owner_db) (which resolves an
    /// unqualified name by search order): the binder maps a `schema.name` reference's
    /// qualifier through here to get the [`DbIndex`] it then fetches the def from.
    ///
    /// Default: the two FIXED built-ins only — `main` and `temp`/`temporary`
    /// ([`DbIndex::from_schema_name`]) — which is correct for any single-namespace store
    /// (a lone `SchemaCatalog` has no attached aliases). Only the composite
    /// [`MultiCatalog`](crate::MultiCatalog) overrides it to also resolve ATTACH aliases,
    /// since only it holds the connection's namespace registry.
    fn db_of_schema(&self, schema: &str) -> Result<Option<DbIndex>> {
        Ok(DbIndex::from_schema_name(schema))
    }

    /// Rebuild the in-memory cache from the `sqlite_schema` b-tree at page 1. Called
    /// when opening a database so the store reflects what is on disk. It is
    /// idempotent — it clears and repopulates the cache — so it may be called again
    /// to resync after an external schema change.
    fn load(&mut self, pager: &dyn Pager) -> Result<()>;

    /// Register a new table (`CREATE TABLE`): allocate its b-tree root, persist its
    /// `sqlite_schema` row on page 1, bump the schema cookie, and cache it. `sql` is
    /// the verbatim `CREATE TABLE` text to store. Errors if a table of the same name
    /// (case-insensitively) already exists, unless `stmt.if_not_exists` is set (then
    /// it is a no-op). Because tables, indexes, views, and triggers share one
    /// namespace, it is also a hard error if the name is already an index, view, or
    /// trigger — `IF NOT EXISTS` suppresses only a same-type (table) collision, never
    /// a cross-type one. Errors on a reserved `sqlite_`-prefixed name.
    fn create_table(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &CreateTable,
        sql: &str,
    ) -> Result<()>;

    /// Register a new index on an existing table (`CREATE INDEX`): allocate its
    /// (empty) b-tree root, persist its `sqlite_schema` row on page 1, bump the schema
    /// cookie, and cache it. `sql` is the verbatim `CREATE INDEX` text to store.
    /// Existing table rows are NOT backfilled into the new index — that is the
    /// executor's job after this returns. Errors if an index of the same name
    /// (case-insensitively) already exists, unless `stmt.if_not_exists` is set (then it
    /// is a no-op); as with `create_table`, `IF NOT EXISTS` suppresses only a same-type
    /// (index) collision, so it is a hard error if the name is already a table, view, or
    /// trigger (the shared namespace). Also errors if the target table does not exist, on
    /// a reserved `sqlite_`-prefixed name, or on an index over an expression rather than
    /// named columns (a deferred gap, not yet supported).
    fn create_index(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &CreateIndex,
        sql: &str,
    ) -> Result<()>;

    /// Register a view (`CREATE VIEW`): persist a `type='view'` row (rootpage 0, the
    /// view has no b-tree; `sql` stored verbatim) on page 1, bump the schema cookie,
    /// and cache it. A view shares the one table/index/view/trigger namespace, so its
    /// name may not collide with any existing object of any of those kinds; as with
    /// `create_table`, `IF NOT EXISTS` suppresses ONLY a same-type (view) duplicate,
    /// never a cross-type collision. Errors on a reserved `sqlite_`-prefixed name.
    ///
    /// A default erroring body is provided so a read-only or test [`Catalog`] need not
    /// implement the write path; only the real store
    /// ([`SchemaCatalog`](crate::SchemaCatalog)) overrides it.
    fn create_view(&mut self, pager: &mut dyn Pager, stmt: &CreateView, sql: &str) -> Result<()> {
        let _ = (pager, stmt, sql);
        Err(minisqlite_types::Error::Sql("CREATE VIEW not supported by this catalog".into()))
    }

    /// Register a trigger (`CREATE TRIGGER`): persist a `type='trigger'` row whose
    /// `tbl_name` is the trigger's TARGET table's bare name (rootpage 0, `sql` stored
    /// verbatim) on page 1, bump the schema cookie, and cache it. Shares the one schema
    /// namespace like every other object; `IF NOT EXISTS` suppresses ONLY a same-type
    /// (trigger) duplicate. Errors on a reserved `sqlite_`-prefixed name or a missing
    /// target table.
    ///
    /// `target` is the engine's resolved verdict about WHERE the `ON`-target lives (see
    /// [`TriggerTarget`](crate::TriggerTarget)). [`SameStore`](crate::TriggerTarget::SameStore)
    /// — the common case — makes the store validate existence and kind against its own
    /// cache exactly as before. [`Foreign`](crate::TriggerTarget::Foreign) — a TEMP trigger
    /// on a `main`/attached object (`lang_createtrigger.html` §7) — carries the already-
    /// validated kind, and the store records the target namespace on the cached
    /// [`TriggerDef`](crate::def::TriggerDef) so fire-time discovery matches the right table.
    ///
    /// A default erroring body is provided, like `create_view`; only the real store
    /// overrides it.
    fn create_trigger(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &CreateTrigger,
        sql: &str,
        target: crate::TriggerTarget,
    ) -> Result<()> {
        let _ = (pager, stmt, sql, target);
        Err(minisqlite_types::Error::Sql("CREATE TRIGGER not supported by this catalog".into()))
    }

    /// Alter an existing table (`ALTER TABLE`): mutate the stored `sqlite_schema`
    /// row(s) on page 1 in place, bump the schema cookie once, and update the cache.
    /// `sql` is the verbatim `ALTER TABLE ...` statement text (the engine passes it),
    /// used to recover the exact column-definition bytes for `ADD COLUMN`.
    ///
    /// All four actions are handled: `ADD COLUMN` appends the column to the stored
    /// `CREATE TABLE` text (existing rows are not rewritten — a short row reads its new
    /// trailing column as NULL); `RENAME TO` renames the table and cascades the new
    /// name into every dependent index row; `RENAME COLUMN` renames the column in the
    /// table definition and every index that references it; and `DROP COLUMN` removes
    /// the column definition, physically rewrites every row to purge the column's data,
    /// and enforces the §5 restrictions that keep the schema valid (it fails if the
    /// column is a PRIMARY KEY, is UNIQUE, is indexed, or is referenced by a surviving
    /// CHECK / FOREIGN KEY / generated-column expression). `DROP COLUMN` on a `WITHOUT
    /// ROWID` table is refused with a loud error (a documented deferral — its rows are
    /// PK key-records needing a different rewrite primitive). Errors mirror SQLite's:
    /// `no such table`, `no such column`, `duplicate column name`, the `ADD COLUMN`
    /// restriction messages, a shared-namespace collision on `RENAME TO`, and a reserved
    /// `sqlite_` name.
    ///
    /// KNOWN GAP (unchecked — silent, NOT a loud error): references from TRIGGERS and
    /// VIEWS are not inspected. SQLite §3 renames a column inside every dependent
    /// trigger/view, and §5 refuses a `DROP COLUMN` the column of a trigger/view still
    /// names; this catalog does neither, so such an alter silently succeeds and leaves
    /// the trigger/view dangling (it breaks only when later re-parsed/expanded). This is
    /// beyond the enumerated restriction set and needs a scope-aware binder to resolve
    /// which table a reference belongs to — a routed follow-up for when triggers/views
    /// become executable.
    ///
    /// A default erroring body is provided (like `create_view` / `create_trigger`) so
    /// a read-only or test [`Catalog`] need not implement the write path; only the
    /// real store ([`SchemaCatalog`](crate::SchemaCatalog)) overrides it.
    fn alter_table(&mut self, pager: &mut dyn Pager, stmt: &AlterTable, sql: &str) -> Result<()> {
        let _ = (pager, stmt, sql);
        Err(minisqlite_types::Error::Sql("ALTER TABLE not supported by this catalog".into()))
    }

    /// Drop a schema object (`DROP {TABLE|INDEX|VIEW|TRIGGER} [IF EXISTS] name`):
    /// remove its `sqlite_schema` row(s) from page 1, bump the schema cookie, and
    /// update the cache. Like `create_*` it runs inside the caller's transaction and
    /// does not begin/commit/rollback the pager itself, and it caches last, so a
    /// failed persistence never leaves the cache and disk disagreeing.
    ///
    /// `DROP TABLE` cascades exactly like real SQLite: it removes the table's own row
    /// plus every index (auto and explicit) and every trigger whose `tbl_name` is that
    /// table, reading page 1 directly so a loaded-from-disk schema cascades as
    /// faithfully as a freshly-created one. It refuses an internal `sqlite_`-prefixed
    /// table (`table X may not be dropped`). `DROP INDEX` removes one explicit index and
    /// refuses an auto-index — any `sqlite_`-prefixed index name — with SQLite's message
    /// (`index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped`).
    /// `DROP VIEW` / `DROP TRIGGER` remove a matching page-1 row of that type when one is
    /// present and free the object's cached name, so the shared namespace no longer
    /// reports it. The page-1 row is found by scanning page 1 directly, so a
    /// loaded-from-disk view/trigger drops as faithfully as a freshly-created one.
    ///
    /// A missing object errors (`no such table/index/view/trigger: name`) unless
    /// `stmt.if_exists` is set, in which case the call is a silent no-op that writes
    /// nothing (no row delete, no cookie bump). Names match case-insensitively, like
    /// every other lookup.
    ///
    /// This frees no b-tree pages: deleting the `sqlite_schema` row(s) leaks the dropped
    /// object's data/index pages until a freelist reclaim path exists. A leaked page
    /// never affects query correctness (the b-tree layer's documented stance).
    fn drop_object(&mut self, pager: &mut dyn Pager, stmt: &Drop) -> Result<()>;
}
