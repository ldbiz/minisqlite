//! `SqlEngine` — the concrete [`Engine`](crate::Engine) the facade routes through.
//! It owns the one instance of each component seam (planner, executor, catalog,
//! pager) plus the per-connection [`Runtime`], and answers a SQL program by
//! parsing it once and running each statement in order (see [`crate::dispatch`],
//! [`crate::txn`], [`crate::pragma`] for the per-kind handling).
//!
//! Everything here is wiring: real planning lives in `minisqlite-plan`, real
//! execution in `minisqlite-exec`, real schema/storage in `minisqlite-catalog` /
//! `minisqlite-pager`. The engine never re-implements them — it routes.

use std::path::{Path, PathBuf};

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, MultiCatalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
// The page size for a freshly created database is the on-disk format's own default
// (4096); reuse the codec's constant so the two cannot drift.
use minisqlite_fileformat::DEFAULT_PAGE_SIZE;
use minisqlite_pager::{DiskPager, MemPager, Pager};
use minisqlite_plan::{Plan, QueryPlanner};
use minisqlite_sql::parse;
use minisqlite_types::{DbIndex, NamespaceMeta, QueryResult, Result, Row};

use crate::Engine;

/// One database connection: the owned component seams plus the connection's
/// transaction state. Fields are `pub(crate)` so the sibling dispatch modules
/// ([`crate::dispatch`] / [`crate::txn`] / [`crate::pragma`]) can drive them; they
/// are not part of any public surface (only the [`Engine`] trait is).
pub struct SqlEngine {
    /// The per-namespace stores, indexed by [`DbIndex`](minisqlite_types::DbIndex):
    /// `main` is ALWAYS index 0, `temp` (created lazily on the first temp object) is
    /// index 1, attached databases take 2.. . Held as a `Vec` (not a single pager) so
    /// the executor can borrow the whole slice as a namespace-aware [`PagerSet`] and a
    /// DML write can reborrow one target store mutably.
    ///
    /// Kept as SEPARATE parallel vecs with [`SqlEngine::catalogs`] (not a `Vec` of
    /// combined structs) so `&self.catalogs` (to build a `MultiCatalog`) and
    /// `&mut self.pagers` (the write set) borrow DISJOINTLY, mirroring the disjoint
    /// `&catalog` / `&mut pager` the single-store engine relied on. Every access uses
    /// `self.pagers[db]`; `self.pagers[0]` (main) is always present (an invariant the
    /// constructors uphold), so it never panics.
    pub(crate) pagers: Vec<Box<dyn Pager>>,
    /// The per-namespace schema stores, parallel to [`SqlEngine::pagers`] (`main` = 0,
    /// `temp` = 1, attached = 2..). The engine mutates the target namespace's concrete
    /// `SchemaCatalog` on DDL and lends the whole slice read-only (as a `MultiCatalog`)
    /// to the planner and executor.
    pub(crate) catalogs: Vec<SchemaCatalog>,
    /// The namespace registry — each store's SQL name and backing file — parallel to
    /// [`SqlEngine::pagers`]/[`SqlEngine::catalogs`] (`main` = 0, `temp` = 1, attached =
    /// 2..). It is the single source of truth for `main`'s backing file (folding the old
    /// `db_path` — see [`SqlEngine::in_memory`]), the alias a schema qualifier resolves
    /// against (see [`SqlEngine::db_of_schema`]), and the `PRAGMA database_list` rows.
    /// Held SEPARATE from pagers/catalogs (like they are from each other) so a read of the
    /// names does not borrow the pager write set.
    pub(crate) namespaces: Vec<NamespaceMeta>,
    /// The query planner (owns the built-in function registry); shared read-only
    /// across statements.
    pub(crate) planner: QueryPlanner,
    /// The streaming executor (a unit struct; state lives in the cursor + runtime).
    pub(crate) executor: StreamingExecutor,
    /// Per-connection runtime: change counters, `last_insert_rowid`, the RNG, and
    /// bound parameters. Exactly one per connection, threaded through every pull.
    pub(crate) rt: Runtime,
    /// `true` while an explicit `BEGIN` transaction is open. A transaction may ALSO
    /// be started implicitly by a `SAVEPOINT` in autocommit; that leaves this `false`
    /// but pushes onto `savepoints`, so "a transaction is active" is
    /// [`SqlEngine::txn_active`] (this flag OR a non-empty savepoint stack), not this
    /// flag alone. When neither holds, the connection is in autocommit and each
    /// data/schema change runs in its own implicit transaction.
    pub(crate) in_explicit_txn: bool,
    /// The open savepoints, innermost last, each a `(name, pager-depth)` pair. The
    /// engine owns the NAME -> depth mapping (the pager addresses savepoints by
    /// depth); `RELEASE`/`ROLLBACK TO` resolve a name to the MOST RECENT matching
    /// savepoint, so duplicate names shadow. Non-empty only within a transaction; a
    /// full `COMMIT`/`ROLLBACK` (explicit or implicit) clears it.
    pub(crate) savepoints: Vec<(String, usize)>,
    /// Whether a USER temp object has ever been materialized (a `CREATE TEMP …` /
    /// `temp.`-qualified object). `PRAGMA database_list` reports the `temp` namespace only
    /// once this is set — matching real SQLite, which lists `temp` only after its schema is
    /// materialized, NOT merely because the store exists. The `temp` store is ALSO created
    /// as a side effect of the first `ATTACH` (to reserve index 1 so attached stores land
    /// at 2..); that reservation must NOT make an unused `temp` appear in `database_list`,
    /// which is exactly why this flag is separate from [`SqlEngine::temp_present`]. Sticky:
    /// once `temp` is materialized it stays listed (its backing persists across a rollback
    /// of the object that created it), so this is never cleared.
    pub(crate) temp_user_materialized: bool,
}

impl SqlEngine {
    /// Open a transient in-memory database: a fresh in-memory pager formatted with
    /// an empty `sqlite_schema` (page 1) and an empty schema cache.
    pub fn open_in_memory() -> Result<SqlEngine> {
        let mut pager = MemPager::new(DEFAULT_PAGE_SIZE);
        // A brand-new in-memory database is always fresh, so it always needs
        // formatting. `init_database` writes page 1 through the pager's autocommit
        // path (no explicit transaction needed), matching the exec-crate tests.
        init_database(&mut pager)?;
        let catalog = SchemaCatalog::new();
        Ok(SqlEngine::from_parts(Box::new(pager), catalog, None))
    }

    /// Open (creating if absent) the on-disk database at `path`. A fresh/empty file
    /// is formatted with an empty `sqlite_schema`; an existing file's schema is
    /// loaded from its page 1 so the cache reflects what is on disk.
    pub fn open(path: &Path) -> Result<SqlEngine> {
        let mut pager = DiskPager::open(path)?;
        // `page_count() == 0` is the fresh-file signal (a new DiskPager starts
        // empty). An existing SQLite file already has page 1, so we must NOT
        // re-format it; we load its schema instead.
        if pager.page_count()? == 0 {
            init_database(&mut pager)?;
        }
        let mut catalog = SchemaCatalog::new();
        // Rebuild the cache from page 1. On a just-formatted file this is an empty
        // schema (equivalent to `new()`); on an existing file it recovers every
        // table and index real sqlite wrote.
        catalog.load(&pager)?;
        Ok(SqlEngine::from_parts(Box::new(pager), catalog, Some(path.to_path_buf())))
    }

    /// Assemble an engine over a formatted pager and a loaded catalog, in
    /// autocommit with a fresh runtime. The single place the component seams are
    /// wired together, so both constructors agree on the initial state. `db_path` is
    /// `main`'s backing file (`None` for the transient in-memory backing); it is stored as
    /// `namespaces[0].file`, the one source of truth for `in_memory` (see
    /// [`SqlEngine::in_memory`]).
    fn from_parts(
        pager: Box<dyn Pager>,
        catalog: SchemaCatalog,
        db_path: Option<PathBuf>,
    ) -> SqlEngine {
        SqlEngine {
            // `main` is always namespace 0. `temp` (index 1) is created lazily on the
            // first temp object OR reserved on the first ATTACH; a fresh connection has
            // only `main`, so the multi-store machinery reduces to the single-store hot
            // path. The registry starts with `main`'s name + backing file, index-aligned.
            pagers: vec![pager],
            catalogs: vec![catalog],
            namespaces: vec![NamespaceMeta::new("main", db_path)],
            planner: QueryPlanner::new(),
            executor: StreamingExecutor,
            rt: Runtime::new(),
            in_explicit_txn: false,
            savepoints: Vec::new(),
            temp_user_materialized: false,
        }
    }

    /// Whether this is a transient in-memory database (no backing file for `main`), derived
    /// from `namespaces[0].file` so the two cannot drift. SQLite reports such a database's
    /// `journal_mode` as `memory` and ignores attempts to change it, so [`crate::pragma`]
    /// short-circuits `PRAGMA journal_mode` for it rather than writing WAL/rollback
    /// version bytes into a header no reopen will ever see.
    pub(crate) fn in_memory(&self) -> bool {
        self.namespaces[0].file.is_none()
    }

    /// Resolve a schema QUALIFIER (`main`/`temp`/`temporary` or an ATTACH alias) to its
    /// [`DbIndex`] within THIS connection's registry, or `None` for an unknown qualifier.
    /// This is the engine-side twin of [`MultiCatalog::db_of_schema`](minisqlite_catalog::Catalog::db_of_schema)
    /// (both delegate to the one [`NamespaceMeta::resolve`] rule), used by the DDL routers
    /// in [`crate::namespace`] to send `alias.tbl` DDL to the attached store.
    pub(crate) fn db_of_schema(&self, schema: &str) -> Option<DbIndex> {
        NamespaceMeta::resolve(&self.namespaces, schema)
    }

    /// Apply `PRAGMA page_size = N` (pragma.html "page_size", fileformat2 §1.3.2).
    ///
    /// SQLite fixes the page size when the database is CREATED: the pragma only takes
    /// effect while the database is still empty (before any table/data page exists);
    /// once a table has been created it is a silent no-op that still reports the
    /// current size on the GET form. An out-of-range or non-power-of-two `N` is
    /// likewise ignored (SQLite does not error). So this is a no-op unless `N` is a
    /// power of two in `[512, 65536]`, the database still holds only the freshly
    /// formatted page 1, and `N` actually differs from the current size.
    ///
    /// When it does apply, the pager is REBUILT at `N`-byte pages (see
    /// [`SqlEngine::reinit_pager_with_page_size`]) rather than patching the header
    /// byte alone: the off-16 page-size the header records MUST equal the size the
    /// pager physically uses for every page, or the file real sqlite reads back would
    /// be malformed.
    pub(crate) fn set_page_size(&mut self, requested: i64) -> Result<()> {
        self.set_page_size_of(DbIndex::MAIN, requested)
    }

    /// Apply `PRAGMA [schema.]page_size = N` to the RESOLVED database `db` (pragma.html
    /// "page_size", fileformat2 §1.3.2). Same rule as `main`, per database file: it only
    /// takes effect while that database is still empty (page 1 alone) and `N` is a power
    /// of two in `[512, 65536]`; otherwise it is a silent no-op. `set_page_size` above is
    /// the `main`-only shorthand (used by the VACUUM path); this is the namespace-aware
    /// form the `page_size` pragma routes through so `PRAGMA aux.page_size = N` resizes
    /// `aux`, not `main`.
    pub(crate) fn set_page_size_of(&mut self, db: DbIndex, requested: i64) -> Result<()> {
        // Inside a transaction this is a silent no-op: the reconfigure below rebuilds
        // the pager, which would strand the open transaction on the discarded pager and
        // leave the connection inconsistent. SQLite likewise ignores `page_size` inside
        // a transaction, so only reconfigure in true autocommit.
        if self.txn_active() {
            return Ok(());
        }
        // Validate BEFORE touching the pager: `MemPager::new` panics on an invalid
        // page size, so an out-of-range request must be filtered here into a no-op.
        let n = match u32::try_from(requested) {
            Ok(n) if (512..=65_536).contains(&n) && n.is_power_of_two() => n,
            _ => return Ok(()),
        };
        // Only while the database is empty (page 1 alone) and the size truly changes.
        // A fresh db has exactly page 1 from `init_database`; the first CREATE TABLE
        // allocates page 2, so `page_count() > 1` means "a table/data page exists". A
        // resolvable-but-not-live namespace (only `temp` before its first object) is the
        // fresh-empty default: one page at DEFAULT_PAGE_SIZE, so we compare against that
        // WITHOUT materializing temp for a no-op request.
        let (page_count, current_size) = if self.namespace_live(db) {
            (self.pagers[db.index()].page_count()?, self.pagers[db.index()].page_size())
        } else {
            (1, DEFAULT_PAGE_SIZE)
        };
        if page_count != 1 || current_size == n {
            return Ok(());
        }
        // Materialize a not-live temp on demand so the rebuild indexes a live store.
        self.ensure_header_target_live(db)?;
        self.reinit_pager_with_page_size(db, n)
    }

    /// Rebuild database `db`'s (empty) pager so every page is `n` bytes, then reload
    /// its (empty) schema cache from the reformatted page 1.
    ///
    /// - In-memory (`temp`, a `:memory:` attach, an in-memory `main`): swap in a fresh
    ///   [`MemPager`] at `n` and re-format it.
    /// - On-disk: the disk backing has no page-size-taking constructor — it reads the
    ///   size from the page-1 header at open — so a freshly-formatted `n`-byte page 1
    ///   (built via a seed in-memory pager + `init_database`, the one canonical
    ///   formatter) is written over the file and the pager is reopened, which adopts
    ///   `n` from that header. The database is empty, so discarding and recreating the
    ///   file loses no data.
    ///
    /// NOTE: the reformatted page 1 is the default rollback-journal header (version 1),
    /// so a `PRAGMA journal_mode = WAL` issued on an empty db *before* `page_size` is
    /// reset to rollback here. That ordering is untested and outside this crate's
    /// on-disk-format scope (rollback is always set before `page_size` in practice);
    /// documented rather than silently diverging. `page_size` after a table exists is
    /// already a no-op, so an established-schema WAL db is never reset.
    fn reinit_pager_with_page_size(&mut self, db: DbIndex, n: u32) -> Result<()> {
        // Precondition (enforced by the sole caller `set_page_size_of`): `n` is a
        // power of two in [512, 65536], and `db` is live (materialized just above).
        // `MemPager::new` panics otherwise, so pin it here for any future second caller.
        debug_assert!(
            (512..=65_536).contains(&n) && n.is_power_of_two(),
            "reinit_pager_with_page_size requires a validated page size, got {n}",
        );
        debug_assert!(
            self.namespace_live(db),
            "reinit_pager_with_page_size requires a live namespace, got index {}",
            db.index()
        );
        let i = db.index();
        match self.namespaces[i].file.clone() {
            None => {
                let mut pager = MemPager::new(n);
                init_database(&mut pager)?;
                self.pagers[i] = Box::new(pager);
            }
            Some(path) => {
                // Build the canonical formatted page 1 for size `n` in memory, then
                // seed the file with it and reopen. The prior on-disk handle is
                // dropped when `self.pagers[i]` is reassigned below; positioned I/O means
                // the interim overwrite does not disturb it.
                let page1 = {
                    let mut seed = MemPager::new(n);
                    init_database(&mut seed)?;
                    seed.read_page(1)?.to_vec()
                };
                std::fs::write(&path, &page1)?;
                self.pagers[i] = Box::new(DiskPager::open(&path)?);
            }
        }
        self.catalogs[i].load(&*self.pagers[i])?;
        Ok(())
    }

    /// Run a whole SQL program: parse it once, then run each statement in order,
    /// returning the result set of the LAST statement that produced one (a `SELECT`
    /// or a row-producing `PRAGMA`). Statements before it run for their side
    /// effects. `None` means no statement produced a result set (e.g. only DDL).
    ///
    /// A statement error propagates immediately: earlier statements' side effects
    /// have already committed in their own implicit transactions (matching SQLite's
    /// per-statement autocommit), and any implicit transaction of the failing
    /// statement was already rolled back at the point of failure.
    pub(crate) fn run_program(&mut self, sql: &str) -> Result<Option<QueryResult>> {
        let ast = parse(sql)?;
        // Invariant (documented on `Ast`): sources runs parallel to statements.
        debug_assert_eq!(ast.statements.len(), ast.statement_sources.len());
        let mut last: Option<QueryResult> = None;
        for (stmt, source) in ast.statements.iter().zip(ast.statement_sources.iter()) {
            if let Some(result) = self.run_statement(stmt, source)? {
                last = Some(result);
            }
            // FULL auto_vacuum truncates the freelist off the file at every commit — but
            // that compaction now happens ATOMICALLY inside the pager's own commit
            // (`minisqlite_pager`'s `av_commit`), in the same durable transaction as the
            // user's data, so there is no separate post-commit sweep to run here. Once a
            // statement has settled to autocommit, reload any catalog whose roots just moved:
            // a root-creating DDL (or an in-commit reclaim) relocates roots to the front
            // (§1.8 roots-first), rewriting `sqlite_schema.rootpage`, so the next statement
            // must resolve the new page, not the stale pre-move one. Skipped while an
            // explicit transaction is still open (roots move at its COMMIT, not
            // mid-transaction), and a cheap per-namespace bool poll otherwise.
            if !self.txn_active() {
                self.reload_catalogs_after_root_move()?;
            }
        }
        Ok(last)
    }

    /// Execute a compiled plan and collect its rows.
    ///
    /// This must stay a single function: it borrows four disjoint fields at once —
    /// `executor` (`&mut`), `catalogs` (`&`, lent as a `MultiCatalog`), and `pagers`
    /// (`&mut`, lent as a `PagerSet`) for the cursor's life, and `rt` (`&mut`) on each
    /// pull. Splitting it behind a `&mut self` helper would borrow all of `self` at
    /// once and stop compiling. The `MultiCatalog` resolves an unqualified name in
    /// SQLite search order (temp, main, attached) and a `*_in(db)` lookup to the named
    /// store; a base plan node's `db` picks the pager the cursor opens on. With only
    /// `main` live this is the single-store path exactly. Transaction wrapping for a
    /// mutating plan is the caller's job (see [`crate::dispatch`]).
    pub(crate) fn execute_plan_collect(&mut self, plan: &Plan) -> Result<Vec<Row>> {
        let catalog = MultiCatalog::new(&self.catalogs, &self.namespaces);
        let mut cursor =
            self.executor.execute(plan, &catalog, PagerSet::Set(&mut self.pagers))?;
        let mut rows = Vec::new();
        while let Some(row) = cursor.next_row(&mut self.rt)? {
            rows.push(row);
        }
        Ok(rows)
    }
}

impl Engine for SqlEngine {
    fn execute(&mut self, sql: &str) -> Result<()> {
        // Run every statement for its side effects; discard any result rows.
        self.run_program(sql).map(|_| ())
    }

    fn query(&mut self, sql: &str) -> Result<QueryResult> {
        // A program with no result-producing statement yields an empty result set,
        // as SQLite does for a statement that returns no rows.
        self.run_program(sql)
            .map(|opt| opt.unwrap_or_else(|| QueryResult { columns: Vec::new(), rows: Vec::new() }))
    }
}
