//! `VACUUM ... INTO <filename>` — write a fresh, defragmented copy of the current
//! database into a new file WITHOUT modifying the source (lang_vacuum.html §3).
//!
//! The copy is built correct-by-construction rather than page-cloned: replay the
//! source's `CREATE` statements onto a fresh target, then copy each table's rows by
//! their EXACT on-disk record bytes and rowid. Because the record codec and b-tree
//! are shared between source and target, a byte-exact cell copy is lossless for every
//! type (INTEGER/REAL/TEXT/BLOB/NULL) and for overflowing values, and it preserves
//! the rowid — the fidelity bar required to match real sqlite. The
//! result is a valid database real sqlite reads back with identical logical content;
//! it is NOT required to be byte-identical to sqlite's own VACUUM output (page layout
//! may differ), only content-identical.
//!
//! FIDELITY-OR-FAIL: any object this path cannot copy faithfully aborts the whole
//! statement with a loud error. A GAP precondition (WITHOUT ROWID, a UTF-16 source)
//! is checked BEFORE the target is even created, so a rejected VACUUM never touches
//! the filesystem. An UNEXPECTED error after creation (an I/O fault mid-copy) discards
//! the partial file on the way out (see `cleanup_partial_target`), so a failed VACUUM
//! INTO always leaves nothing behind and is retryable — a marked gap, never a silent
//! lie or an orphaned half-copy.
//!
//! MEMORY: the copy streams. Row data is committed to the target in bounded chunks
//! (`COMMIT_CHUNK_BYTES`) rather than one transaction over the whole database, because
//! the pager stages every written page in `Txn.overlay` until commit — a single
//! transaction would make peak RSS grow with the DB size. Chunked commits hold peak
//! target WRITE memory to ~O(max(chunk, largest single row)), independent of how large
//! the vacuumed database is. (Two residual peaks are engine-wide, not VACUUM-specific,
//! so they are out of this slice: a file-backed SOURCE grows the pager's read cache with
//! the source size as the cursor scans it, and each index backfill stages one whole
//! index — the same as a normal `CREATE INDEX`.)

use std::path::Path;

use minisqlite_btree::{table_insert, TableCursor};
use minisqlite_catalog::{is_reserved_schema_name, Catalog, SchemaRow};
use minisqlite_fileformat::{decode_record_enc, TextEncoding};
use minisqlite_pager::{text_encoding_of, PageId, Pager};
use minisqlite_sql::{Expr, Literal};
use minisqlite_types::{DbIndex, Error, Result};

use crate::engine::SqlEngine;

/// Page 1 is always the `sqlite_schema` b-tree root (fileformat2 §1.3), so the
/// source schema is read by scanning the b-tree rooted here.
const SCHEMA_ROOT: PageId = 1;

/// Commit the target's data transaction and re-begin once this many payload bytes have
/// been staged since the last commit. Every written page lives in the pager's
/// copy-on-write `Txn.overlay` until commit (there is no spill), so this bounds peak
/// target memory to ~O(chunk) instead of O(database). The bound is measured in logical
/// record bytes (`payload.len()`), a within-a-constant-factor proxy for the staged page
/// footprint, and a single row larger than this is still staged whole (its post-insert
/// check trips the next commit), so the true bound is O(max(chunk, largest row)). Chosen
/// to keep the fsync count small relative to the bytes moved while staying flat in RSS: a
/// few MiB is the same order of magnitude as real sqlite's VACUUM page buffer.
const COMMIT_CHUNK_BYTES: usize = 4 * 1024 * 1024;

impl SqlEngine {
    /// Execute `VACUUM INTO <into>`: copy this database to the file named by `into`.
    ///
    /// Returns `Ok(())` on success (the caller maps it to `Ok(None)` — VACUUM produces
    /// no rows). The source pager, catalog, and transaction state are only READ, never
    /// mutated, so the connection is unchanged afterward.
    ///
    /// Ordering matters for atomicity of the gaps: every "cannot copy this faithfully"
    /// precondition is checked BEFORE the target file is created, so a rejected VACUUM
    /// never leaves a half-written output.
    pub(crate) fn exec_vacuum_into(&mut self, into: &Expr) -> Result<()> {
        // --- Preconditions (all before creating the target) ---

        // VACUUM cannot run inside a transaction (lang_vacuum.html): it is itself a
        // top-level operation. `txn_active` is the same "explicit BEGIN or open
        // savepoint" test `set_page_size` guards on.
        if self.txn_active() {
            return Err(Error::sql("cannot VACUUM from within a transaction"));
        }

        let filename = eval_into_filename(into)?;
        ensure_target_absent_or_empty(&filename)?;

        // The general record codec is now encoding-aware, but this VACUUM INTO copy path
        // does NOT yet thread the source/target encoding through `copy_all_table_data`
        // (it replays DDL and copies record cells without transcoding), so a UTF-16 source
        // would leave the target header's encoding and its freshly written schema rows
        // disagreeing — a corrupt mix. Until the copy path threads `enc`, refuse a UTF-16
        // source loudly rather than emit a wrong file (regular in-place writes to a UTF-16
        // database ARE supported; only this whole-db copy is not yet).
        let src_enc = text_encoding_of(&*self.pagers[0]);
        if src_enc != TextEncoding::Utf8 {
            return Err(Error::sql(
                "VACUUM INTO: only UTF-8 source databases are supported \
                 (the copy path does not yet thread UTF-16 encoding)",
            ));
        }
        let src_page_size = self.pagers[0].page_size();
        // The source's page count bounds a valid b-tree root: an internal table's root is
        // taken from its raw `sqlite_schema.rootpage` (below), so it is range-checked
        // against this before being cast to a `PageId` (see `checked_source_root`).
        let src_page_count = self.pagers[0].page_count()?;

        // The source schema in rowid (creation) order: this gives verbatim `CREATE`
        // text and a dependency-safe order (a table precedes its indexes/triggers).
        let schema = read_source_schema(&*self.pagers[0], src_enc)?;

        // WITHOUT ROWID tables are keyed by their PRIMARY KEY, not an integer rowid, so
        // the rowid+payload cell copy below cannot reproduce them. Detect and reject
        // before creating the target (honest, tested gap) rather than emit a table with
        // the wrong physical shape.
        for row in &schema {
            if is_user_table(row) && self.source_table_without_rowid(&row.name)? {
                return Err(Error::sql(format!(
                    "VACUUM INTO: WITHOUT ROWID table {:?} is not yet supported",
                    row.name
                )));
            }
        }

        // --- Create the target, then populate it (discard the file on any failure) ---

        // `open` formats a fresh file with an empty schema (page 1 only). Everything from
        // here on writes to that file, so a failure part-way leaves a partial `.db`; the
        // guard below removes it so a failed VACUUM INTO leaves nothing behind and is
        // retryable to the same path (matching real sqlite, and making the module's
        // "no partial/wrong file" invariant literally true for unexpected errors too).
        let target_path = Path::new(&filename);
        let mut target = SqlEngine::open(target_path)?;
        let result = self.vacuum_populate(&mut target, src_page_size, src_page_count, &schema);
        // Release the target's file handle before touching the path on the error cleanup.
        drop(target);
        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                cleanup_partial_target(target_path);
                Err(e)
            }
        }
    }

    /// Fill the freshly-created `target` from this (source) database. Split from the
    /// preconditions so `exec_vacuum_into` can wrap the whole post-creation span in one
    /// discard-on-error guard. Reads the source only (`&self`).
    ///
    /// Phase order matters and each phase owns its own transaction(s):
    /// 1. Match the source page size (the target is still empty here).
    /// 2. Replay `CREATE TABLE` (each an autocommit txn) so a table and its auto-indexes
    ///    exist before its rows land.
    /// 3. Copy row data in bounded chunks (`copy_table_rows` commits periodically).
    /// 4. Backfill the (still-empty) auto-indexes from the committed rows.
    /// 5. Carry the internal bookkeeping tables' contents (`sqlite_sequence`/`_stat1`).
    /// 6. Replay `CREATE INDEX`/`VIEW`/`TRIGGER` — explicit indexes build from the rows
    ///    now present, triggers register without firing on the low-level row copy.
    fn vacuum_populate(
        &self,
        target: &mut SqlEngine,
        src_page_size: u32,
        src_page_count: PageId,
        schema: &[SchemaRow],
    ) -> Result<()> {
        // Matching the page size keeps the copy faithful and is cheap; content is correct
        // either way.
        target.set_page_size(src_page_size as i64)?;

        // Internal `sqlite_*` rows are skipped here: they auto-create (sqlite_sequence
        // when an AUTOINCREMENT table is replayed) or are ensured explicitly
        // (sqlite_stat1) during the data copy.
        for row in schema {
            if is_user_table(row) {
                if let Some(sql) = &row.sql {
                    target.run_program(sql)?;
                }
            }
        }

        copy_all_table_data(&*self.pagers[0], &self.catalogs[0], target, schema)?;
        backfill_all_auto_indexes(target, schema)?;
        copy_sqlite_sequence(&*self.pagers[0], target, schema, src_page_count)?;
        copy_sqlite_stat1(&*self.pagers[0], target, schema, src_page_count)?;

        for row in schema {
            if is_user_object(row) && matches!(row.obj_type.as_str(), "index" | "view" | "trigger") {
                if let Some(sql) = &row.sql {
                    target.run_program(sql)?;
                }
            }
        }

        Ok(())
    }

    /// Whether the SOURCE table `name` is a WITHOUT ROWID table, read from the source's
    /// own (already-loaded, kept-current) catalog. A schema row with no matching catalog
    /// entry is corruption, surfaced rather than silently treated as rowid-keyed.
    fn source_table_without_rowid(&self, name: &str) -> Result<bool> {
        let def = self.catalogs[0].table(name)?.ok_or_else(|| {
            Error::format(format!("VACUUM INTO: source table {name:?} missing from catalog"))
        })?;
        Ok(def.without_rowid)
    }
}

/// Copy every user table's rows (byte-exact) into the matching target table. Each table
/// is streamed and re-inserted by its exact rowid + record bytes via the low-level
/// `table_insert`, which does NOT maintain indexes — the auto-indexes stay empty here
/// and are backfilled next. The SOURCE root comes from the source's VALIDATED catalog
/// (not the raw `sqlite_schema.rootpage`), so a bad root was already rejected at open.
fn copy_all_table_data(
    src_pager: &dyn Pager,
    src_catalog: &dyn Catalog,
    target: &mut SqlEngine,
    schema: &[SchemaRow],
) -> Result<()> {
    for row in schema {
        if is_user_table(row) {
            let source_root = src_catalog
                .table(&row.name)?
                .ok_or_else(|| {
                    Error::format(format!(
                        "VACUUM INTO: source table {:?} missing from catalog",
                        row.name
                    ))
                })?
                .root_page;
            let target_root = target
                .catalogs[0]
                .table(&row.name)?
                .ok_or_else(|| {
                    Error::format(format!(
                        "VACUUM INTO: target table {:?} missing after schema replay",
                        row.name
                    ))
                })?
                .root_page;
            copy_table_rows(src_pager, &mut *target.pagers[0], source_root, target_root)?;
        }
    }
    Ok(())
}

/// Copy every row of one table b-tree into another, preserving the exact rowid and the
/// exact record payload bytes. Streams one row at a time (a large table never
/// materializes). `cursor.payload()` reassembles any overflow chain, and `table_insert`
/// re-splits it for the target's page size, so overflowing values copy losslessly even
/// across a page-size change.
///
/// The copy is COMMITTED IN CHUNKS (`COMMIT_CHUNK_BYTES`), managing the target's write
/// transactions itself: begin lazily on the first row, commit + re-begin each time the
/// staged payload exceeds the budget, commit the tail. This keeps the pager's overlay —
/// and thus peak memory — bounded regardless of table size (the source is read-only, so
/// re-beginning between chunks is always safe). On any error a still-open chunk is
/// rolled back so the target pager is left clean (the whole file is discarded anyway).
fn copy_table_rows(
    src_pager: &dyn Pager,
    target_pager: &mut dyn Pager,
    source_root: PageId,
    target_root: PageId,
) -> Result<()> {
    match copy_table_rows_chunked(src_pager, target_pager, source_root, target_root) {
        Ok(()) => Ok(()),
        Err(e) => {
            // A chunk transaction may still be open (error mid-chunk). Discard it; on a
            // clean chunk boundary there is no active txn and `rollback` errors, which we
            // intentionally ignore — the copy error is what propagates.
            let _ = target_pager.rollback();
            Err(e)
        }
    }
}

fn copy_table_rows_chunked(
    src_pager: &dyn Pager,
    target_pager: &mut dyn Pager,
    source_root: PageId,
    target_root: PageId,
) -> Result<()> {
    let mut cursor = TableCursor::open(src_pager, source_root)?;
    let mut positioned = cursor.first()?;
    let mut open_txn = false;
    let mut pending_bytes: usize = 0;
    while positioned {
        if !open_txn {
            target_pager.begin()?;
            open_txn = true;
        }
        let rowid = cursor.rowid();
        // The source cell and the target write touch DIFFERENT pagers, so the payload can
        // be passed straight through without an owning copy — an inline row stays a borrow
        // of the source page (no per-row allocation). The `cursor` borrow ends before
        // `next()` because `payload` is scoped to this block.
        let written = {
            let payload = cursor.payload()?;
            table_insert(target_pager, target_root, rowid, &payload)?;
            payload.len()
        };
        pending_bytes += written;
        positioned = cursor.next()?;
        if pending_bytes >= COMMIT_CHUNK_BYTES {
            target_pager.commit()?;
            open_txn = false;
            pending_bytes = 0;
        }
    }
    if open_txn {
        target_pager.commit()?;
    }
    Ok(())
}

/// Backfill the auto-indexes of every user table from the rows just copied.
fn backfill_all_auto_indexes(target: &mut SqlEngine, schema: &[SchemaRow]) -> Result<()> {
    for row in schema {
        if is_user_table(row) {
            backfill_auto_indexes(target, &row.name)?;
        }
    }
    Ok(())
}

/// Backfill every index currently defined on `table` (the auto-indexes from its
/// UNIQUE/PRIMARY KEY constraints — explicit indexes do not exist yet) from the table's
/// now-copied rows. An auto-index keys named columns only, so its key-expression slice
/// is empty AND it is never PARTIAL (a UNIQUE/PRIMARY KEY constraint has no WHERE clause), so
/// its partial predicate is `None`; the table's GENERATED-column programs are still needed for
/// an auto-index over a generated column (its value is computed, not stored). Explicit indexes
/// — including PARTIAL ones — are recreated later by replaying their `CREATE INDEX` SQL
/// (`run_program`), which threads the predicate through the normal `build_index` dispatch.
///
/// Each index builds in its OWN transaction: `build_index` runs a full-table scan and
/// stages the whole index in the overlay, so committing per index keeps that bounded to
/// one index (the same shape a real `CREATE INDEX` has) rather than adding to the data
/// copy's peak.
fn backfill_auto_indexes(target: &mut SqlEngine, table: &str) -> Result<()> {
    // VACUUM operates entirely on the concrete main store (`catalogs[0]`/`pagers[0]`), so the
    // namespace is `MAIN`; over a concrete single-namespace catalog `table_in` is the bare
    // lookup, so this is behavior-identical to the pre-`db` form.
    let generated =
        target.planner.table_generated_programs(&target.catalogs[0], DbIndex::MAIN, table)?;
    let index_names: Vec<String> =
        target.catalogs[0].indexes_on(table)?.iter().map(|ix| ix.name.clone()).collect();
    for name in &index_names {
        target.pagers[0].begin()?;
        match minisqlite_exec::build_index(
            &mut *target.pagers[0],
            &target.catalogs[0],
            name,
            &generated,
            &[],
            None,
        ) {
            Ok(()) => target.pagers[0].commit()?,
            Err(e) => {
                let _ = target.pagers[0].rollback();
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Copy `sqlite_sequence` data if the source has one AND the target created one. The
/// target's `sqlite_sequence` auto-creates only when an AUTOINCREMENT table is replayed;
/// if the source carries an (orphaned) empty one the target never made, there is nothing
/// to preserve.
fn copy_sqlite_sequence(
    src_pager: &dyn Pager,
    target: &mut SqlEngine,
    schema: &[SchemaRow],
    src_page_count: PageId,
) -> Result<()> {
    let Some(src_row) = find_internal_table(schema, "sqlite_sequence") else {
        return Ok(());
    };
    let target_root = match target.catalogs[0].table("sqlite_sequence")? {
        Some(def) => def.root_page,
        None => return Ok(()),
    };
    let source_root = checked_source_root(src_row.rootpage, src_page_count)?;
    copy_table_rows(src_pager, &mut *target.pagers[0], source_root, target_root)
}

/// Copy `sqlite_stat1` data if the source has one. Unlike `sqlite_sequence` it does not
/// auto-create (no ANALYZE runs during the copy), so its schema/root is ensured first —
/// in its own transaction, committed before the (chunked) row copy re-begins.
fn copy_sqlite_stat1(
    src_pager: &dyn Pager,
    target: &mut SqlEngine,
    schema: &[SchemaRow],
    src_page_count: PageId,
) -> Result<()> {
    let Some(src_row) = find_internal_table(schema, "sqlite_stat1") else {
        return Ok(());
    };
    let source_root = checked_source_root(src_row.rootpage, src_page_count)?;
    target.pagers[0].begin()?;
    let target_root = match target.catalogs[0].ensure_sqlite_stat1(&mut *target.pagers[0]) {
        Ok(root) => {
            target.pagers[0].commit()?;
            root
        }
        Err(e) => {
            let _ = target.pagers[0].rollback();
            return Err(e);
        }
    };
    copy_table_rows(src_pager, &mut *target.pagers[0], source_root, target_root)
}

/// Range-check a raw `sqlite_schema.rootpage` before casting it to a `PageId`. A b-tree
/// root is always page `>= 2` (page 1 is `sqlite_schema`) and within the file, so a
/// negative / zero / one / oversized value is corrupt schema — reject it at the boundary
/// rather than let `as PageId` silently truncate into a bogus page that only fails later
/// on a page read (the fail-closed convention the catalog's rootpage load uses). Used for
/// the internal tables, whose roots are read raw rather than via the validated catalog.
fn checked_source_root(rootpage: i64, page_count: PageId) -> Result<PageId> {
    if rootpage >= 2 && rootpage <= i64::from(page_count) {
        Ok(rootpage as PageId)
    } else {
        Err(Error::format(format!(
            "VACUUM INTO: source object has out-of-range root page {rootpage}"
        )))
    }
}

/// Best-effort removal of a partially-written target and its journal/WAL sidecars after
/// a mid-copy failure, so a failed `VACUUM INTO` leaves nothing behind. Errors are
/// ignored: the copy already failed and that error is what the caller returns.
fn cleanup_partial_target(path: &Path) {
    let _ = std::fs::remove_file(path);
    let base = path.as_os_str();
    for suffix in ["-journal", "-wal", "-shm"] {
        let mut sidecar = base.to_os_string();
        sidecar.push(suffix);
        let _ = std::fs::remove_file(std::path::PathBuf::from(sidecar));
    }
}

/// Read the source's full `sqlite_schema` in rowid order by scanning the page-1 b-tree
/// directly (a normal query would work too, but this yields the verbatim `sql` and the
/// on-disk root pages without going through the planner).
fn read_source_schema(pager: &dyn Pager, enc: TextEncoding) -> Result<Vec<SchemaRow>> {
    let mut cursor = TableCursor::open(pager, SCHEMA_ROOT)?;
    let mut out = Vec::new();
    let mut positioned = cursor.first()?;
    while positioned {
        let payload = cursor.payload()?;
        out.push(SchemaRow::from_values(&decode_record_enc(&payload, enc))?);
        positioned = cursor.next()?;
    }
    Ok(out)
}

/// A user (non-internal) table row. Internal `sqlite_*` objects are not replayed as
/// DDL; their b-trees are recreated on demand and only their DATA is copied.
fn is_user_table(row: &SchemaRow) -> bool {
    row.obj_type == "table" && !is_reserved_schema_name(&row.name)
}

/// A user (non-internal) schema object of any kind.
fn is_user_object(row: &SchemaRow) -> bool {
    !is_reserved_schema_name(&row.name)
}

/// Find an internal bookkeeping TABLE row by name (case-insensitive), e.g.
/// `sqlite_sequence` / `sqlite_stat1`.
fn find_internal_table<'a>(schema: &'a [SchemaRow], name: &str) -> Option<&'a SchemaRow> {
    schema.iter().find(|r| r.obj_type == "table" && r.name.eq_ignore_ascii_case(name))
}

/// Evaluate the `INTO` expression to a filename. Only a constant string is accepted
/// (a bare string literal, or a single parenthesized one). A non-constant expression
/// is rejected loudly rather than guessed at — the facade binds no parameters here, so
/// there is no runtime value to resolve.
fn eval_into_filename(into: &Expr) -> Result<String> {
    match into {
        Expr::Literal(Literal::Text(s)) => Ok(s.clone()),
        Expr::Parenthesized(items) if items.len() == 1 => eval_into_filename(&items[0]),
        _ => Err(Error::sql("VACUUM INTO target must be a constant string filename")),
    }
}

/// The target must not already exist as a NON-EMPTY file (lang_vacuum.html §3): an
/// absent path or a zero-length file is acceptable (the copy fills it), anything else
/// errors WITHOUT touching the file, so an existing database is never clobbered.
fn ensure_target_absent_or_empty(filename: &str) -> Result<()> {
    match std::fs::metadata(filename) {
        Ok(meta) if meta.len() > 0 => Err(Error::sql("output file already exists")),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::io(format!("VACUUM INTO: cannot stat target file: {e}"))),
    }
}
