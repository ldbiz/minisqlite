//! `MultiCatalog` — a read-only composite [`Catalog`] over one connection's live schema
//! stores (`main`, `temp`, attached), presented as a single catalog to the planner and
//! executor.
//!
//! It is NOT a second seam: it implements the SAME [`Catalog`] trait (the seam guard
//! permits many impls, only one trait). The engine keeps the concrete per-namespace
//! [`SchemaCatalog`]s in a `Vec` and, for each read (planning + execution), lends a
//! `MultiCatalog` borrowing that slice. DDL/writes never flow through here — the engine
//! mutates the target namespace's concrete `SchemaCatalog` directly — so the write
//! methods are `unreachable!`.
//!
//! ## Two lookup modes
//! * The BARE methods ([`table`](Catalog::table), …) run SQLite's name-resolution SEARCH
//!   ORDER — temp first (when a temp store exists), then main, then attached — returning
//!   the first match. This is what an UNQUALIFIED name resolves to, and it is why a temp
//!   table shadows a same-named main table.
//! * The `*_in(db, …)` methods target ONE explicit namespace. The binder stamps the
//!   resolved [`DbIndex`] onto each plan node, and the executor looks the def up with
//!   `*_in` so a schema-qualified / shadowed name opens its cursor on the RIGHT store.
//!
//! ## Index convention (load-bearing)
//! `main` is always index 0 and `temp`, once it exists, is always index 1 (attached take
//! 2..). So a connection with no temp/attached store is a one-element slice `[main]`, for
//! which every bare method and every `*_in(MAIN, …)` reduces to `main`'s own lookup — the
//! single-store hot path is byte-for-byte unchanged.

use crate::catalog::Catalog;
use crate::def::{IndexDef, TableDef, TriggerDef, TriggerTargetDb, ViewDef};
use crate::schemacatalog::SchemaCatalog;
use minisqlite_pager::Pager;
use minisqlite_sql::{AlterTable, CreateIndex, CreateTable, CreateTrigger, CreateView, Drop};
use minisqlite_types::{DbIndex, NamespaceMeta, Result};

/// A borrowed, read-only view over the connection's live [`SchemaCatalog`]s, indexed by
/// [`DbIndex`] (`main` = 0, `temp` = 1, attached = 2..). Built fresh per statement from
/// the engine's owning `Vec`, so it never copies a schema.
///
/// It also borrows the connection's namespace registry (`namespaces`, index-aligned with
/// `catalogs`) so it can resolve a schema QUALIFIER — including an ATTACH alias — to its
/// [`DbIndex`] via [`db_of_schema`](Catalog::db_of_schema). The registry is names only for
/// resolution; the def bytes still come from `catalogs`.
pub struct MultiCatalog<'a> {
    catalogs: &'a [SchemaCatalog],
    namespaces: &'a [NamespaceMeta],
}

impl<'a> MultiCatalog<'a> {
    /// Wrap the connection's live schema stores and their namespace registry. `catalogs[0]`
    /// MUST be `main` and `namespaces` MUST be index-aligned with `catalogs` (the engine
    /// upholds both); a one-element pair is the common no-temp/no-attach case.
    pub fn new(catalogs: &'a [SchemaCatalog], namespaces: &'a [NamespaceMeta]) -> Self {
        debug_assert!(!catalogs.is_empty(), "MultiCatalog needs at least the main store at index 0");
        debug_assert_eq!(
            catalogs.len(),
            namespaces.len(),
            "MultiCatalog namespaces must be index-aligned with catalogs",
        );
        Self { catalogs, namespaces }
    }

    /// The store indices in SQLite name-resolution search order: temp (index 1, when a
    /// temp store is present), then main (0), then attached (2..) in attach order. For the
    /// common one-store slice this yields just `[0]`.
    fn search_order(&self) -> impl Iterator<Item = usize> {
        let n = self.catalogs.len();
        let temp = (n >= 2).then_some(1);
        temp.into_iter().chain(std::iter::once(0)).chain(2..n)
    }

    /// The search-order index of the first store that DEFINES `name` as a table or view
    /// (the namespace an unqualified reference to `name` resolves to), or `None`.
    fn owner_of(&self, name: &str) -> Result<Option<usize>> {
        for i in self.search_order() {
            if self.catalogs[i].table(name)?.is_some() || self.catalogs[i].view(name)?.is_some() {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    /// Does the CROSS-NAMESPACE temp trigger `t` bind to namespace `db` RIGHT NOW? The
    /// binding is a stable NAME, re-resolved against the live registry every call so it
    /// survives a `DETACH` that renumbers indices:
    /// * [`ForeignSchema(name)`](TriggerTargetDb::ForeignSchema) — the target's schema name
    ///   resolves (via the borrowed registry) to `db`; never names `temp`, so this is `false`
    ///   for `db == TEMP`;
    /// * [`ForeignUnqualified`](TriggerTargetDb::ForeignUnqualified) — the target's bare name
    ///   resolves by search order to `db` (the doc's reattach-on-schema-change behavior); this
    ///   CAN be `db == TEMP` when a shadowing `temp.t` has appeared, which is why
    ///   [`triggers_on_in`](Catalog::triggers_on_in) runs the cross-namespace pass for `TEMP` too.
    /// A [`SameStore`](TriggerTargetDb::SameStore) def never matches here (it is handled by the
    /// home-store branch); callers only pass temp-store triggers, so this is `false` for one.
    ///
    /// The engine mirrors this rule in `SqlEngine::foreign_binds_to` (its DROP-cascade victim
    /// resolution) so create-time DROP and fire-time discovery agree on the SAME set; the two
    /// copies exist only because the engine resolves while holding `&mut self` over its own
    /// `Vec`s. Keep them in lockstep — change one variant's rule and you MUST change the twin.
    fn foreign_binds_to(&self, t: &TriggerDef, db: DbIndex) -> Result<bool> {
        match &t.target {
            TriggerTargetDb::SameStore => Ok(false),
            TriggerTargetDb::ForeignSchema(schema) => {
                Ok(NamespaceMeta::resolve(self.namespaces, schema) == Some(db))
            }
            TriggerTargetDb::ForeignUnqualified => {
                Ok(self.owner_of(&t.table)? == Some(db.index()))
            }
        }
    }
}

impl Catalog for MultiCatalog<'_> {
    // ---- bare lookups: SQLite search order (temp, main, attached; first match) --------

    fn table(&self, name: &str) -> Result<Option<&TableDef>> {
        for i in self.search_order() {
            if let Some(t) = self.catalogs[i].table(name)? {
                return Ok(Some(t));
            }
        }
        Ok(None)
    }

    fn index(&self, name: &str) -> Result<Option<&IndexDef>> {
        for i in self.search_order() {
            if let Some(x) = self.catalogs[i].index(name)? {
                return Ok(Some(x));
            }
        }
        Ok(None)
    }

    fn view(&self, name: &str) -> Result<Option<&ViewDef>> {
        for i in self.search_order() {
            if let Some(v) = self.catalogs[i].view(name)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// The indexes of the table `name` resolves to (an index lives in the same store as
    /// its table), so this returns the indexes of the FIRST search-order store owning
    /// `name` — matching what [`table`](Catalog::table) resolves it to.
    fn indexes_on<'b>(&'b self, table: &str) -> Result<Vec<&'b IndexDef>> {
        for i in self.search_order() {
            if self.catalogs[i].table(table)?.is_some() {
                return self.catalogs[i].indexes_on(table);
            }
        }
        Ok(Vec::new())
    }

    /// The triggers that fire when a write hits the object `table` RESOLVES to by search
    /// order (temp, then main, then attached) — its home-store triggers UNION any TEMP
    /// trigger bound to that exact `(namespace, name)`.
    ///
    /// Delegates to [`triggers_on_in`](Catalog::triggers_on_in) at the resolved owner
    /// namespace, so the cross-namespace union and the shadowing rule (a temp trigger
    /// bound to `main.t` fires on `main.t`, never on a shadowing `temp.t`) are defined in
    /// ONE place. Real firing always goes through `triggers_on_in` with the DML-resolved
    /// `db`; this bare form mirrors it for an unqualified reference.
    fn triggers_on(&self, table: &str) -> Result<Vec<&TriggerDef>> {
        match self.owner_of(table)? {
            Some(i) => self.triggers_on_in(DbIndex(i as u16), table),
            None => Ok(Vec::new()),
        }
    }

    /// The namespace a bare `name` resolves to as a table or view, in search order —
    /// exactly [`owner_of`](MultiCatalog::owner_of) mapped to a [`DbIndex`] (the slice
    /// index IS the namespace index: `catalogs[0]` = `main`, `catalogs[1]` = `temp`). This
    /// is what lets the binder stamp `TEMP` on an unqualified reference to a temp-shadowed
    /// name, so the executor opens its cursor on the temp store.
    fn owner_db(&self, name: &str) -> Result<Option<DbIndex>> {
        Ok(self.owner_of(name)?.map(|i| DbIndex(i as u16)))
    }

    /// The namespace a bare INDEX `name` resolves to, in search order — the first store
    /// that defines `name` as an index (temp, main, attached). Mirrors [`owner_db`] for
    /// indexes so an unqualified `pragma_index_info(ix)` / `PRAGMA index_info(ix)` reads the
    /// same store bare `index(name)` resolves to.
    fn owner_db_of_index(&self, name: &str) -> Result<Option<DbIndex>> {
        for i in self.search_order() {
            if self.catalogs[i].index(name)?.is_some() {
                return Ok(Some(DbIndex(i as u16)));
            }
        }
        Ok(None)
    }

    /// Resolve a schema QUALIFIER to its namespace: the two built-ins (`main`/`temp`) plus
    /// this connection's ATTACH aliases, via [`NamespaceMeta::resolve`] over the borrowed
    /// registry. This is the composite's override of the trait default (built-ins only) —
    /// the one place `aux.tbl` learns that `aux` is store index 2+.
    fn db_of_schema(&self, schema: &str) -> Result<Option<DbIndex>> {
        Ok(NamespaceMeta::resolve(self.namespaces, schema))
    }

    /// Every user table across all live stores. FOREIGN KEY enforcement should prefer
    /// [`tables_in`](Catalog::tables_in) (a cross-database FK is forbidden, so only the
    /// write's own namespace matters); this bare union is the namespace-blind fallback.
    fn tables(&self) -> Result<Vec<&TableDef>> {
        let mut out = Vec::new();
        for c in self.catalogs {
            out.extend(c.tables()?);
        }
        Ok(out)
    }

    // ---- namespace-qualified lookups: route to catalogs[db] ----------------------------
    //
    // An out-of-range `db` (a plan naming a store that is not live) resolves to "absent"
    // rather than panicking: the read fails closed as a normal "no such object", and the
    // binder only ever stamps a `db` for a live store, so the miss is not reachable in
    // practice.

    fn table_in(&self, db: DbIndex, name: &str) -> Result<Option<&TableDef>> {
        match self.catalogs.get(db.index()) {
            Some(c) => c.table(name),
            None => Ok(None),
        }
    }

    fn index_in(&self, db: DbIndex, name: &str) -> Result<Option<&IndexDef>> {
        match self.catalogs.get(db.index()) {
            Some(c) => c.index(name),
            None => Ok(None),
        }
    }

    fn view_in(&self, db: DbIndex, name: &str) -> Result<Option<&ViewDef>> {
        match self.catalogs.get(db.index()) {
            Some(c) => c.view(name),
            None => Ok(None),
        }
    }

    fn indexes_on_in<'b>(&'b self, db: DbIndex, table: &str) -> Result<Vec<&'b IndexDef>> {
        match self.catalogs.get(db.index()) {
            Some(c) => c.indexes_on(table),
            None => Ok(Vec::new()),
        }
    }

    /// Every trigger that fires on a write to the EXACT `(db, table)` — the fire-time
    /// discovery seam the executor's trigger path uses (with the DML-resolved `db`).
    ///
    /// Two sources, unioned:
    /// * HOME-store triggers bound to the same store — `catalogs[db]` triggers whose
    ///   `target.is_same_store()`. For a non-temp store that is every trigger it holds (only
    ///   the temp store ever holds a cross-namespace trigger), so this branch is byte-for-byte
    ///   the old behavior and the common `main`-trigger-on-`main`-table hot path is unchanged.
    ///   For `db == TEMP` the same-store filter is what excludes a temp trigger BOUND
    ///   ELSEWHERE (e.g. to `main.t`) from firing on a shadowing `temp.t`.
    /// * CROSS-NAMESPACE temp triggers bound to `(db, table)` — temp-store foreign triggers
    ///   whose binding RE-RESOLVES to `db` through the connection's live registry (an
    ///   [`ForeignSchema`](crate::TriggerTargetDb::ForeignSchema) name) or search order (an
    ///   [`ForeignUnqualified`](crate::TriggerTargetDb::ForeignUnqualified) target) — see
    ///   [`foreign_binds_to`](MultiCatalog::foreign_binds_to). Resolving by NAME every
    ///   statement (never a cached numeric index) is what keeps a trigger on `aux.u` bound to
    ///   `aux.u` across an unrelated `DETACH` (`lang_createtrigger.html` §7). Gated on
    ///   [`has_cross_namespace_triggers`](SchemaCatalog::has_cross_namespace_triggers) so a
    ///   temp store with none pays nothing (and it is `None` — skipped — when no temp store
    ///   exists, the common single-store hot path).
    ///
    ///   This pass runs for `db == TEMP` TOO, not just non-temp targets: an
    ///   [`ForeignUnqualified`](crate::TriggerTargetDb::ForeignUnqualified) trigger whose
    ///   unqualified target now resolves (by search order) to a SHADOWING `temp.t` has
    ///   "reattached" to it (`lang_createtrigger.html` §7's documented reattach-on-schema-
    ///   change behavior for an unqualified TEMP-trigger target), so it must be discoverable
    ///   on a write to `temp.t`. No double-listing results: a same-store temp trigger is caught
    ///   only by the home pass (`foreign_binds_to` is `false` for it) and a foreign one only
    ///   here (the home pass filters it out), and a `ForeignSchema` binding never names `temp`.
    ///
    /// Ordering is by folded trigger name, matching the concrete store's deterministic
    /// (name-order) firing approximation, so a unioned result stays reproducible run to run.
    /// The home contribution already arrives name-ordered, so the merge sort runs ONLY when a
    /// cross-namespace trigger was actually added.
    fn triggers_on_in(&self, db: DbIndex, table: &str) -> Result<Vec<&TriggerDef>> {
        let mut out: Vec<&TriggerDef> = Vec::new();

        // Home-store triggers bound to (db, table): same-store targets only.
        if let Some(c) = self.catalogs.get(db.index()) {
            out.extend(c.triggers_on(table)?.into_iter().filter(|t| t.target.is_same_store()));
        }

        // Cross-namespace temp triggers bound to (db, table): held in the temp store, each
        // re-resolved by name against the CURRENT registry/search order (so `foreign_binds_to`
        // — never a cached index — decides membership). The `has_cross_namespace_triggers` gate
        // keeps the common "temp store, no foreign trigger" case byte-for-byte the old
        // single-store hot path. Runs for `db == TEMP` too so a reattached `ForeignUnqualified`
        // trigger fires on a shadowing `temp.t` (see the doc-comment note; `foreign_binds_to`
        // returns `false` for a same-store def, so it is never double-listed with the home pass).
        let mut added_foreign = false;
        if let Some(temp) = self.catalogs.get(DbIndex::TEMP.index()) {
            if temp.has_cross_namespace_triggers() {
                for t in temp.triggers_on(table)? {
                    if self.foreign_binds_to(t, db)? {
                        out.push(t);
                        added_foreign = true;
                    }
                }
            }
        }

        // The home contribution is already name-ordered; only a merged cross-namespace
        // trigger can disturb that, so re-sort only then.
        if added_foreign {
            out.sort_by_cached_key(|t| t.name.to_ascii_lowercase());
        }
        Ok(out)
    }

    fn tables_in(&self, db: DbIndex) -> Result<Vec<&TableDef>> {
        match self.catalogs.get(db.index()) {
            Some(c) => c.tables(),
            None => Ok(Vec::new()),
        }
    }

    // ---- write methods: never reached (the engine writes concrete stores directly) -----

    fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
        unreachable!("MultiCatalog is a read-only composite; load a concrete SchemaCatalog")
    }

    fn create_table(&mut self, _p: &mut dyn Pager, _s: &CreateTable, _sql: &str) -> Result<()> {
        unreachable!("MultiCatalog is read-only; CREATE TABLE targets a concrete SchemaCatalog")
    }

    fn create_index(&mut self, _p: &mut dyn Pager, _s: &CreateIndex, _sql: &str) -> Result<()> {
        unreachable!("MultiCatalog is read-only; CREATE INDEX targets a concrete SchemaCatalog")
    }

    fn create_view(&mut self, _p: &mut dyn Pager, _s: &CreateView, _sql: &str) -> Result<()> {
        unreachable!("MultiCatalog is read-only; CREATE VIEW targets a concrete SchemaCatalog")
    }

    fn create_trigger(
        &mut self,
        _p: &mut dyn Pager,
        _s: &CreateTrigger,
        _sql: &str,
        _target: crate::TriggerTarget,
    ) -> Result<()> {
        unreachable!("MultiCatalog is read-only; CREATE TRIGGER targets a concrete SchemaCatalog")
    }

    fn alter_table(&mut self, _p: &mut dyn Pager, _s: &AlterTable, _sql: &str) -> Result<()> {
        unreachable!("MultiCatalog is read-only; ALTER TABLE targets a concrete SchemaCatalog")
    }

    fn drop_object(&mut self, _p: &mut dyn Pager, _s: &Drop) -> Result<()> {
        unreachable!("MultiCatalog is read-only; DROP targets a concrete SchemaCatalog")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_btree::init_database;
    use minisqlite_pager::MemPager;
    use minisqlite_sql::{parse, Statement};

    // --- fixtures --------------------------------------------------------------
    //
    // Each namespace (`main`, `temp`, attached) is its OWN concrete `SchemaCatalog`
    // over its OWN page-1 store — exactly how the engine keeps them; a `MultiCatalog`
    // only BORROWS a slice of them. So a fixture builds each store independently on a
    // fresh pager, then wraps the collected stores. Two same-named objects are given
    // DIFFERENT column sets / SQL so an assertion proves WHICH store answered a lookup
    // (that is the whole point — a search-order bug would return the wrong store's def,
    // not an error).

    /// A pager holding a freshly formatted, empty database (page 1 = empty schema).
    fn fresh_pager() -> MemPager {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        pager
    }

    /// Apply one CREATE statement to `cat` through the real catalog write path (the
    /// same route the engine drives), panicking if it is not a supported CREATE or if
    /// the create fails — a fixture must not silently build a store missing an object.
    fn apply(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        let ast = parse(sql).unwrap();
        match &ast.statements[0] {
            Statement::CreateTable(s) => cat.create_table(pager, s, sql).unwrap(),
            Statement::CreateIndex(s) => cat.create_index(pager, s, sql).unwrap(),
            Statement::CreateView(s) => cat.create_view(pager, s, sql).unwrap(),
            Statement::CreateTrigger(s) => {
                cat.create_trigger(pager, s, sql, crate::TriggerTarget::SameStore).unwrap()
            }
            _ => panic!("unsupported DDL in test fixture: {sql}"),
        }
    }

    /// Build ONE namespace's concrete store by running `ddl` in order on its own pager.
    /// The pager is only needed while writing (a `SchemaCatalog` read answers from its
    /// in-memory cache), so it may drop when this returns — the store is self-contained.
    fn store(ddl: &[&str]) -> SchemaCatalog {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        for sql in ddl {
            apply(&mut cat, &mut pager, sql);
        }
        cat
    }

    /// Build a (temp) store holding ONE cross-namespace foreign-bound trigger, mirroring the
    /// engine's `trigger_target` verdict for a `CREATE TEMP TRIGGER … ON <schema>.<tbl> …`.
    /// `schema` is the `ON`-qualifier as written (`Some` → `ForeignSchema`, `None` →
    /// `ForeignUnqualified`); `is_view` is the (engine-resolved) target kind. This is exactly
    /// the fixture the old NOTE below claimed was un-buildable before the engine wired temp
    /// triggers across schemas — `create_trigger` now accepts the `Foreign` verdict, so it is.
    fn store_with_foreign_trigger(sql: &str, schema: Option<&str>, is_view: bool) -> SchemaCatalog {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let ast = parse(sql).unwrap();
        let Statement::CreateTrigger(s) = &ast.statements[0] else {
            panic!("fixture must be a single CREATE TRIGGER: {sql}");
        };
        let target = crate::TriggerTarget::Foreign { schema: schema.map(str::to_string), is_view };
        cat.create_trigger(&mut pager, s, sql, target).unwrap();
        cat
    }

    /// An index-aligned namespace registry for an `n`-store fixture: `main` (0), `temp`
    /// (1), then attached `aux2`, `aux3`, … (2..). The search-order tests key off POSITION
    /// (not name), so the names only matter to the `db_of_schema` resolution tests below,
    /// which use these exact aliases.
    fn metas(n: usize) -> Vec<NamespaceMeta> {
        (0..n)
            .map(|i| {
                let name = match i {
                    0 => "main".to_string(),
                    1 => "temp".to_string(),
                    k => format!("aux{k}"),
                };
                NamespaceMeta::new(name, None)
            })
            .collect()
    }

    /// The column names of a table def — the fingerprint that says WHICH store an
    /// ambiguous (shadowed) lookup resolved to.
    fn cols(t: &TableDef) -> Vec<String> {
        t.columns.iter().map(|c| c.name.clone()).collect()
    }

    // --- 1: a one-store slice reduces to `main` --------------------------------

    #[test]
    fn one_store_slice_reduces_to_main() {
        // With only `main` present the search order is just `[0]`, so every bare method
        // and `*_in(MAIN, …)` is main's own lookup, and TEMP (index 1) is out of range.
        let stores = [store(&["CREATE TABLE t(a)"])];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert!(mc.table("t").unwrap().is_some(), "bare finds the sole store's table");
        assert!(mc.table("absent").unwrap().is_none(), "a missing name is None");
        assert!(mc.table("T").unwrap().is_some(), "resolution folds ASCII case (T == t)");

        assert!(mc.table_in(DbIndex::MAIN, "t").unwrap().is_some(), "MAIN targets store 0");
        assert!(
            mc.table_in(DbIndex::TEMP, "t").unwrap().is_none(),
            "TEMP is out of range for a one-store slice -> absent, NOT a panic"
        );
    }

    // --- 2: shadowing — the load-bearing case ----------------------------------

    #[test]
    fn bare_lookup_resolves_temp_over_main_under_shadowing() {
        // A temp table `t` shadows a same-named main `t`: a BARE (unqualified) lookup
        // must resolve to temp (searched first), while the qualified `*_in` forms still
        // reach each explicit namespace. Distinct columns ([a] vs [b]) prove the source,
        // so this pins bare != qualified under shadowing.
        let stores = [store(&["CREATE TABLE t(a)"]), store(&["CREATE TABLE t(b)"])];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert_eq!(
            cols(mc.table("t").unwrap().unwrap()),
            ["b"],
            "bare `t` resolves to TEMP (searched first), not MAIN"
        );
        assert_eq!(
            cols(mc.table_in(DbIndex::MAIN, "t").unwrap().unwrap()),
            ["a"],
            "table_in(MAIN) targets main's t"
        );
        assert_eq!(
            cols(mc.table_in(DbIndex::TEMP, "t").unwrap().unwrap()),
            ["b"],
            "table_in(TEMP) targets temp's t"
        );
    }

    // --- 3: a main-only name is found by falling through the search order -------

    #[test]
    fn main_only_name_found_through_search_order() {
        // A name defined only in main is found by the bare search falling THROUGH temp
        // (which holds an unrelated table) to main.
        let stores = [store(&["CREATE TABLE only_main(a)"]), store(&["CREATE TABLE other(b)"])];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert!(mc.table("only_main").unwrap().is_some(), "search falls through temp to main");
        assert!(mc.table_in(DbIndex::MAIN, "only_main").unwrap().is_some());
        assert!(
            mc.table_in(DbIndex::TEMP, "only_main").unwrap().is_none(),
            "the name is not in temp"
        );
    }

    // --- 4: a temp-only name ---------------------------------------------------

    #[test]
    fn temp_only_name_found_in_temp() {
        let stores = [store(&["CREATE TABLE main_t(a)"]), store(&["CREATE TABLE only_temp(b)"])];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert!(mc.table("only_temp").unwrap().is_some(), "found in temp (searched first)");
        assert!(mc.table_in(DbIndex::TEMP, "only_temp").unwrap().is_some());
        assert!(
            mc.table_in(DbIndex::MAIN, "only_temp").unwrap().is_none(),
            "the name is not in main"
        );
    }

    // --- 5: indexes_on / index follow the resolved owner -----------------------

    #[test]
    fn indexes_on_follows_resolved_owner_under_shadowing() {
        // `indexes_on(name)` returns the indexes of the FIRST search-order store that
        // owns `name` as a table — the same store bare `table(name)` resolves to — so
        // under shadowing it reports TEMP's index, while `indexes_on_in(MAIN, …)` reports
        // main's. Distinct index name + columns prove the source.
        let stores = [
            store(&["CREATE TABLE t(a)", "CREATE INDEX im ON t(a)"]),
            store(&["CREATE TABLE t(b)", "CREATE INDEX it ON t(b)"]),
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let bare: Vec<(String, Vec<String>)> = mc
            .indexes_on("t")
            .unwrap()
            .iter()
            .map(|d| (d.name.clone(), d.columns.clone()))
            .collect();
        assert_eq!(
            bare,
            vec![("it".to_string(), vec!["b".to_string()])],
            "bare indexes_on follows the temp-resolved owner"
        );

        let in_main: Vec<(String, Vec<String>)> = mc
            .indexes_on_in(DbIndex::MAIN, "t")
            .unwrap()
            .iter()
            .map(|d| (d.name.clone(), d.columns.clone()))
            .collect();
        assert_eq!(
            in_main,
            vec![("im".to_string(), vec!["a".to_string()])],
            "indexes_on_in(MAIN) targets main's index"
        );
    }

    #[test]
    fn bare_index_lookup_resolves_temp_over_main() {
        // A bare `index(name)` runs the same search order: a same-named index in both
        // stores resolves temp-first, while `index_in` reaches each explicit store.
        let stores = [
            store(&["CREATE TABLE t(a)", "CREATE INDEX ix ON t(a)"]),
            store(&["CREATE TABLE t(b)", "CREATE INDEX ix ON t(b)"]),
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert_eq!(mc.index("ix").unwrap().unwrap().columns, ["b"], "bare index -> temp");
        assert_eq!(
            mc.index_in(DbIndex::MAIN, "ix").unwrap().unwrap().columns,
            ["a"],
            "index_in(MAIN) -> main"
        );
        assert_eq!(mc.index_in(DbIndex::TEMP, "ix").unwrap().unwrap().columns, ["b"]);
    }

    // --- 6: view / view_in shadowing -------------------------------------------

    #[test]
    fn view_shadowing_bare_vs_qualified() {
        // Mirror the table-shadowing case for views: a temp view `v` shadows a main `v`;
        // bare resolves temp-first, `view_in` targets the named store. The stored
        // verbatim SQL differs, so it proves which store answered.
        let main_sql = "CREATE VIEW v AS SELECT a FROM base";
        let temp_sql = "CREATE VIEW v AS SELECT b FROM base";
        let stores = [
            store(&["CREATE TABLE base(a)", main_sql]),
            store(&["CREATE TABLE base(b)", temp_sql]),
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert_eq!(mc.view("v").unwrap().unwrap().sql, temp_sql, "bare view -> temp");
        assert_eq!(
            mc.view_in(DbIndex::MAIN, "v").unwrap().unwrap().sql,
            main_sql,
            "view_in(MAIN) -> main's view"
        );
        assert_eq!(mc.view_in(DbIndex::TEMP, "v").unwrap().unwrap().sql, temp_sql);
    }

    // --- 7: triggers_on resolves via the owning store --------------------------

    #[test]
    fn triggers_on_resolves_via_owning_store() {
        // A trigger lives in the same store as its target. With target `t` shadowed,
        // bare `triggers_on("t")` reports the temp-resolved owner's trigger (via
        // `owner_of`), while `triggers_on_in(db, …)` targets the named store.
        let stores = [
            store(&["CREATE TABLE t(a)", "CREATE TRIGGER tm AFTER INSERT ON t BEGIN SELECT 1; END"]),
            store(&["CREATE TABLE t(b)", "CREATE TRIGGER tt AFTER INSERT ON t BEGIN SELECT 1; END"]),
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let bare: Vec<String> =
            mc.triggers_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(bare, ["tt"], "bare triggers_on -> temp-resolved owner's trigger");

        let in_main: Vec<String> = mc
            .triggers_on_in(DbIndex::MAIN, "t")
            .unwrap()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(in_main, ["tm"], "triggers_on_in(MAIN) -> main's trigger");

        let in_temp: Vec<String> = mc
            .triggers_on_in(DbIndex::TEMP, "t")
            .unwrap()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(in_temp, ["tt"], "triggers_on_in(TEMP) -> temp's trigger");
    }

    // --- 8: tables() unions all stores; tables_in targets one -------------------

    #[test]
    fn tables_unions_all_stores_while_tables_in_targets_one() {
        // `tables()` is the namespace-blind union of every store's user tables;
        // `tables_in(db)` is only the named store's. Order is unspecified (a HashMap
        // walk), so sort before comparing. Both exclude internal `sqlite_*` tables per
        // the SchemaCatalog contract, so only the user tables are counted.
        let stores = [
            store(&["CREATE TABLE m1(a)", "CREATE TABLE m2(a)"]),
            store(&["CREATE TABLE tp1(b)"]),
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let mut all: Vec<String> = mc.tables().unwrap().iter().map(|d| d.name.clone()).collect();
        all.sort();
        assert_eq!(all, ["m1", "m2", "tp1"], "tables() unions main + temp user tables");

        let mut in_main: Vec<String> =
            mc.tables_in(DbIndex::MAIN).unwrap().iter().map(|d| d.name.clone()).collect();
        in_main.sort();
        assert_eq!(in_main, ["m1", "m2"], "tables_in(MAIN) is main only");

        let in_temp: Vec<String> =
            mc.tables_in(DbIndex::TEMP).unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(in_temp, ["tp1"], "tables_in(TEMP) is temp only");
    }

    // --- 9: an out-of-range namespace is absent, never a panic -----------------

    #[test]
    fn out_of_range_namespace_is_absent_not_panic() {
        // A `db` past the live stores fails closed (None / empty vec) via the checked
        // `catalogs.get(db.index())`, never an index panic — for EVERY `*_in` variant.
        let stores = [store(&[
            "CREATE TABLE t(a)",
            "CREATE INDEX ix ON t(a)",
            "CREATE VIEW v AS SELECT a FROM t",
            "CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END",
        ])];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);
        let far = DbIndex(9);

        assert!(mc.table_in(far, "t").unwrap().is_none());
        assert!(mc.index_in(far, "ix").unwrap().is_none());
        assert!(mc.view_in(far, "v").unwrap().is_none());
        assert!(mc.indexes_on_in(far, "t").unwrap().is_empty());
        assert!(mc.triggers_on_in(far, "t").unwrap().is_empty());
        assert!(mc.tables_in(far).unwrap().is_empty());

        // The SAME objects ARE reachable through their real namespace (MAIN = 0), so the
        // emptiness above is the range check firing, not a lookup into an empty store.
        assert!(mc.table_in(DbIndex::MAIN, "t").unwrap().is_some());
        assert!(mc.index_in(DbIndex::MAIN, "ix").unwrap().is_some());
        assert!(mc.view_in(DbIndex::MAIN, "v").unwrap().is_some());
        assert_eq!(mc.tables_in(DbIndex::MAIN).unwrap().len(), 1);
    }

    // --- 10: the attached tail of the search order (store index 2..) ------------

    #[test]
    fn attached_store_is_searched_after_temp_and_main() {
        // A three-store slice `[main, temp, attached]` is the only shape that reaches
        // the `2..n` tail of `search_order` (temp(1), main(0), then attached(2..)). This
        // pins the FULL precedence chain temp > main > attached, which the 1/2-store
        // cases cannot: in particular that attached is searched AFTER main (a regression
        // that walked `2..n` before main would resolve a main/attached collision to the
        // attached store). Distinct columns ([a] main, [b] temp, [c] attached) prove the
        // answering store.
        let stores = [
            store(&["CREATE TABLE t(a)", "CREATE TABLE u(a)", "CREATE TABLE only_main(a)"]), // main (0)
            store(&["CREATE TABLE t(b)"]),                                                   // temp (1)
            store(&["CREATE TABLE t(c)", "CREATE TABLE u(c)", "CREATE TABLE only_att(c)"]),  // attached (2)
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);
        let attached = DbIndex(2);

        // `t` is in all three -> temp (searched first) wins.
        assert_eq!(cols(mc.table("t").unwrap().unwrap()), ["b"], "temp (1) shadows main and attached");
        // `u` is in main + attached only -> MAIN wins, proving attached is searched
        // strictly AFTER main (else this would be [c]).
        assert_eq!(cols(mc.table("u").unwrap().unwrap()), ["a"], "main (0) beats attached (2)");
        // an attached-only name is found ONLY by walking into the `2..n` tail.
        assert_eq!(cols(mc.table("only_att").unwrap().unwrap()), ["c"], "search falls through to attached");
        // a main-only name is unaffected by the extra store.
        assert!(mc.table("only_main").unwrap().is_some());

        // The qualified form targets each explicit namespace, including attached (2).
        assert_eq!(cols(mc.table_in(DbIndex::TEMP, "t").unwrap().unwrap()), ["b"]);
        assert_eq!(cols(mc.table_in(DbIndex::MAIN, "u").unwrap().unwrap()), ["a"]);
        assert_eq!(cols(mc.table_in(attached, "t").unwrap().unwrap()), ["c"], "table_in(2) -> attached");
        assert_eq!(cols(mc.table_in(attached, "u").unwrap().unwrap()), ["c"]);
        assert!(mc.table_in(attached, "only_main").unwrap().is_none(), "only_main is not in attached");
    }

    #[test]
    fn all_lookup_methods_reach_the_attached_tail() {
        // Every bare lookup family (`table`/`index`/`view`) and every owner-resolving
        // family (`indexes_on`, `triggers_on` via `owner_of`) walks the SAME search order,
        // so each must reach the attached tail. Here the attached store (index 2) is the
        // sole home of `only_att` and its dependent index/view/trigger, while main and
        // temp hold unrelated objects — so a lookup that stopped before `2..n` would miss.
        let stores = [
            store(&["CREATE TABLE m(a)"]),  // main (0): unrelated
            store(&["CREATE TABLE tp(b)"]), // temp (1): unrelated
            store(&[
                "CREATE TABLE only_att(c)",
                "CREATE INDEX att_ix ON only_att(c)",
                "CREATE VIEW att_v AS SELECT c FROM only_att",
                "CREATE TRIGGER att_trg AFTER INSERT ON only_att BEGIN SELECT 1; END",
            ]), // attached (2)
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);
        let attached = DbIndex(2);

        assert!(mc.table("only_att").unwrap().is_some(), "table() reaches attached");
        assert!(mc.index("att_ix").unwrap().is_some(), "index() reaches attached");
        assert!(mc.view("att_v").unwrap().is_some(), "view() reaches attached");

        let ix: Vec<String> =
            mc.indexes_on("only_att").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(ix, ["att_ix"], "indexes_on() resolves an attached-owned table");

        let trg: Vec<String> =
            mc.triggers_on("only_att").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(trg, ["att_trg"], "triggers_on()/owner_of() resolve an attached-owned table");

        // The qualified forms target index 2 directly.
        assert!(mc.index_in(attached, "att_ix").unwrap().is_some());
        assert!(mc.view_in(attached, "att_v").unwrap().is_some());
        assert_eq!(
            mc.indexes_on_in(attached, "only_att").unwrap().iter().map(|d| d.name.clone()).collect::<Vec<_>>(),
            ["att_ix"]
        );
        assert_eq!(
            mc.triggers_on_in(attached, "only_att").unwrap().iter().map(|d| d.name.clone()).collect::<Vec<_>>(),
            ["att_trg"]
        );
    }

    // --- 11: owner_of resolves a VIEW owner (its `|| view(name)` branch) --------

    #[test]
    fn triggers_on_resolves_a_view_owner_via_owner_of() {
        // A trigger's target can be a VIEW (INSTEAD OF), so `owner_of` resolves table OR
        // view. Case 7 uses a TABLE target, where a table match alone satisfies owner_of
        // and its view branch never runs. Here `vw` exists ONLY as a view (no same-named
        // table anywhere), so the trigger is reachable ONLY through owner_of's view
        // branch — dropping `|| view(name).is_some()` would make this go red.
        let stores = [
            store(&["CREATE TABLE base(a)"]), // main (0): unrelated
            store(&[
                "CREATE TABLE base(b)",
                "CREATE VIEW vw AS SELECT b FROM base",
                "CREATE TRIGGER vt INSTEAD OF INSERT ON vw BEGIN SELECT 1; END",
            ]), // temp (1): the view + its INSTEAD OF trigger
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let bare: Vec<String> =
            mc.triggers_on("vw").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(bare, ["vt"], "triggers_on resolves a VIEW owner via owner_of's view branch");

        let in_temp: Vec<String> =
            mc.triggers_on_in(DbIndex::TEMP, "vw").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(in_temp, ["vt"], "triggers_on_in(TEMP) targets the view's own store");
    }

    // --- 11b: cross-namespace TEMP trigger union + shadowing (no longer a gap) --

    #[test]
    fn triggers_on_in_unions_and_shadows_a_foreign_schema_temp_trigger() {
        // `SchemaCatalog::create_trigger` now accepts a `TriggerTarget::Foreign` verdict, so a
        // temp store CAN hold a trigger bound to `main.t` — the very fixture an earlier NOTE
        // said was un-buildable. `triggers_on_in(MAIN, "t")` must UNION it (it fires on main.t),
        // while `triggers_on_in(TEMP, "t")` must NOT (a shadowing temp.t is a different
        // (db, name); the trigger is bound to main, not temp).
        let stores = [
            store(&["CREATE TABLE t(a)", "CREATE TABLE audit(v)"]), // main (0)
            store_with_foreign_trigger(
                "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t BEGIN SELECT 1; END",
                Some("main"),
                false,
            ), // temp (1): a trigger bound to main.t (ForeignSchema)
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let on_main: Vec<String> =
            mc.triggers_on_in(DbIndex::MAIN, "t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(on_main, ["trg"], "the cross-namespace temp trigger fires on main.t");

        let on_temp: Vec<String> =
            mc.triggers_on_in(DbIndex::TEMP, "t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert!(
            on_temp.is_empty(),
            "it must NOT fire on a shadowing temp.t (bound to main, not temp): {on_temp:?}"
        );
    }

    #[test]
    fn triggers_on_in_reattaches_a_foreign_unqualified_temp_trigger_to_a_shadowing_temp_table() {
        // An UNQUALIFIED foreign binding re-resolves by SEARCH ORDER every call, so a shadowing
        // `temp.t` makes it "reattach" to temp.t (lang_createtrigger.html §7's documented
        // reattach-on-schema-change behavior): `triggers_on_in(TEMP, "t")` finds it, and
        // `triggers_on_in(MAIN, "t")` no longer does (moved, not duplicated).
        let stores = [
            store(&["CREATE TABLE t(a)"]), // main (0): the original resolved target
            {
                // temp (1): a ForeignUnqualified trigger on bare `t`, PLUS a shadowing temp.t.
                let mut pager = fresh_pager();
                let mut cat = SchemaCatalog::new();
                let sql = "CREATE TEMP TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END";
                let ast = parse(sql).unwrap();
                let Statement::CreateTrigger(s) = &ast.statements[0] else { unreachable!() };
                cat.create_trigger(
                    &mut pager,
                    s,
                    sql,
                    crate::TriggerTarget::Foreign { schema: None, is_view: false },
                )
                .unwrap();
                apply(&mut cat, &mut pager, "CREATE TABLE t(a)"); // temp.t now shadows main.t
                cat
            },
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let on_temp: Vec<String> =
            mc.triggers_on_in(DbIndex::TEMP, "t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(on_temp, ["trg"], "reattaches to the shadowing temp.t");

        let on_main: Vec<String> =
            mc.triggers_on_in(DbIndex::MAIN, "t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert!(on_main.is_empty(), "no longer fires on main.t once temp.t shadows it: {on_main:?}");
    }

    // --- 12: tables() is a namespace-blind union (keeps shadowed duplicates) ----

    #[test]
    fn tables_union_keeps_shadowed_duplicates() {
        // Unlike the single-object bare lookups (which return the FIRST match), `tables()`
        // is a namespace-BLIND union of every store's tables (its doc comment) and does
        // NOT dedupe a shadowed name: a table present in both stores appears TWICE. Pin
        // this so an accidental future de-dup — which would silently change the meaning of
        // the FK-enforcement fallback — is caught.
        let stores = [store(&["CREATE TABLE t(a)"]), store(&["CREATE TABLE t(b)"])];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        let names: Vec<String> = mc.tables().unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(names.len(), 2, "the union spans both stores");
        assert_eq!(
            names.iter().filter(|n| n.as_str() == "t").count(),
            2,
            "a shadowed name is NOT deduped by the namespace-blind union"
        );
    }

    // --- 13: db_of_schema resolves the two built-ins + attached aliases --------

    #[test]
    fn db_of_schema_resolves_builtins_and_attached_aliases() {
        // The qualified-resolution override: `main`/`temp`/`temporary` map to the fixed
        // built-ins (case-insensitively), an attached alias maps to its store index (here
        // `aux2` = index 2), and an unknown qualifier is `None` (the binder turns that into
        // "no such table"). Distinct columns prove the alias reaches the RIGHT store when
        // then fetched via `table_in`.
        let stores = [
            store(&["CREATE TABLE m(a)"]),           // main (0)
            store(&["CREATE TABLE tp(b)"]),          // temp (1)
            store(&["CREATE TABLE only_att(c)"]),    // attached (2), alias "aux2"
        ];
        let ns = metas(stores.len());
        let mc = MultiCatalog::new(&stores, &ns);

        assert_eq!(mc.db_of_schema("main").unwrap(), Some(DbIndex::MAIN));
        assert_eq!(mc.db_of_schema("MAIN").unwrap(), Some(DbIndex::MAIN));
        assert_eq!(mc.db_of_schema("temp").unwrap(), Some(DbIndex::TEMP));
        assert_eq!(mc.db_of_schema("temporary").unwrap(), Some(DbIndex::TEMP));
        assert_eq!(mc.db_of_schema("aux2").unwrap(), Some(DbIndex(2)));
        assert_eq!(mc.db_of_schema("AUX2").unwrap(), Some(DbIndex(2)), "aliases fold case");
        assert_eq!(mc.db_of_schema("nope").unwrap(), None, "an unknown qualifier is None");

        // The resolved index reaches the alias's own store (its distinct column proves it).
        let db = mc.db_of_schema("aux2").unwrap().unwrap();
        assert_eq!(cols(mc.table_in(db, "only_att").unwrap().unwrap()), ["c"]);
    }

    // --- optional: a write method on the composite is unreachable --------------

    #[test]
    #[should_panic(expected = "read-only")]
    fn write_method_on_composite_is_unreachable() {
        // The composite is READ-ONLY: the engine writes a concrete SchemaCatalog
        // directly, so every write method here is `unreachable!`. Pin that invariant so
        // a future edit that tries to route a write through the composite fails loud.
        // The `expected` substring ties the panic to the real unreachable message, not
        // to any panic in the setup below.
        let stores = [store(&["CREATE TABLE t(a)"])];
        let ns = metas(stores.len());
        let mut mc = MultiCatalog::new(&stores, &ns);
        let mut pager = fresh_pager();
        let ast = parse("CREATE TABLE z(a)").unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else {
            unreachable!("test setup: CREATE TABLE parses to a CreateTable statement");
        };
        let _ = mc.create_table(&mut pager, stmt, "CREATE TABLE z(a)");
    }
}
