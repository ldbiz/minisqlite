//! `ANALYZE` — gather table/index statistics into the internal `sqlite_stat1`
//! table (`spec/sqlite-doc/lang_analyze.html`, `fileformat2.html` §2.6.4).
//!
//! This is additive and safe: it only WRITES `sqlite_stat1`. The planner does not
//! read it, so a query returns byte-identical rows with or without ANALYZE — the
//! observable effect is purely the presence and content of `sqlite_stat1`.
//!
//! ## What a stat row holds (fileformat2 §2.6.4)
//! One row per index: `tbl` = table name, `idx` = index name, and `stat` a string
//! `"N a1 a2 ... aK"` for a K-column index — `N` = rows in the index, and `a_m` =
//! the estimated average number of rows that share the same value across the
//! left-most `m` columns. A table with NO index instead gets a single `idx = NULL`
//! row whose `stat` is just the table's row count, so the planner still knows its
//! size. An empty index/table emits no row, but `ANALYZE` on an empty database
//! still CREATES the (empty) `sqlite_stat1`.
//!
//! ## How the numbers are computed
//! `a_m = (N + nDistinct_m / 2) / nDistinct_m` where `nDistinct_m` is the number of
//! distinct values of the left-most `m` columns: round-to-nearest integer division of
//! the spec's "estimated average" `N / nDistinct_m`. This reading makes the last
//! integer of a unique index exactly 1 (there `nDistinct_K == N`), the one hard
//! constraint `fileformat2` §2.6.4 states. §2.6.4 calls the value "approximate" and
//! gives no exact formula, so this is a spec-correct APPROXIMATION, not a reproduction
//! of any real-SQLite internal: a ceiling reading `(N + nDistinct - 1) / nDistinct`
//! also satisfies the unique→1 rule and would differ on cases like `N=5, D=4` — an
//! honest residual, since the docs do not pin which rounding SQLite uses.
//!
//! The distinct counts are found by STREAMING the b-tree in key order, holding only
//! the previous key and `K` running change counters — O(K) memory, no materialized
//! copy of the index or table (the streaming perf discipline). Keys are compared
//! with [`Collation::Binary`], the order the b-tree itself is built in, so the
//! distinct-prefix boundaries this scan sees are exactly the b-tree's. A NULL counts
//! as a value DISTINCT from every other, INCLUDING another NULL (see
//! [`first_diff_col`]): SQLite legitimately stores multiple NULLs in a UNIQUE index,
//! and §2.6.4 states the unique→last-integer-1 rule unconditionally, so duplicate
//! NULLs must each be their own distinct prefix for that invariant to hold.
//!
//! ## WITHOUT ROWID tables (fileformat2 §2.6.4: idx == tbl)
//! A WITHOUT ROWID table stores its rows in its PRIMARY KEY b-tree (`TableDef.root_page`
//! IS that index — see `minisqlite-exec`'s `without_rowid`), and §2.6.4 records its
//! stat row with `idx` EQUAL to `tbl`: the ordinary index stat over that PK b-tree, with
//! `K` = the number of PRIMARY KEY columns. This engine emits that row (the PK is
//! unique, so its last integer is 1), IN ADDITION to any secondary-index rows. The `idx`
//! value is the table name — fully derivable from the docs — so no name is fabricated.
//! The PK-column count comes from the single authority `TableDef::primary_key` (see
//! [`wr_pk_key_col_count`]), so EVERY WR PK form emits its row — including a table-level
//! single INTEGER PRIMARY KEY, the one shape the older spec+flag derivation missed.
//! Secondary indexes on a WITHOUT ROWID table are not creatable in this engine yet
//! (CREATE INDEX rejects a WR base table), so that path is present but currently
//! unexercised.

use std::cmp::Ordering;

use minisqlite_btree::{table_delete, table_insert, IndexCursor, TableCursor};
use minisqlite_catalog::{is_reserved_schema_name, Catalog, IndexDef, SchemaCatalog, TableDef};
use minisqlite_fileformat::{
    decode_record_enc, decode_record_into_enc, encode_record_enc, TextEncoding,
};
use minisqlite_pager::{text_encoding_of, PageId, Pager};
use minisqlite_sql::QualifiedName;
use minisqlite_types::{compare_values, Collation, Error, Result, Value};

/// Run one `ANALYZE [target]` inside the caller's write transaction (the engine
/// wraps this in `with_write_txn`, so a failure rolls back and leaves no trace).
///
/// Order matters for correctness:
/// 1. Resolve the scope FIRST (a read of the catalog). An unknown target errors
///    here, before anything is written — so `ANALYZE bogus` creates nothing, the
///    way SQLite resolves the name at prepare time before opening `sqlite_stat1`.
/// 2. Ensure `sqlite_stat1` exists (created on the first `ANALYZE` that needs it).
/// 3. Compute fresh stats by streaming the target b-trees (reads only).
/// 4. Replace: delete the in-scope `sqlite_stat1` rows, then insert the fresh ones.
pub(crate) fn run(
    cat: &mut SchemaCatalog,
    pager: &mut dyn Pager,
    target: Option<&QualifiedName>,
) -> Result<()> {
    let plan = resolve(cat, target)?;
    // A pure no-op scope (`DeleteScope::None`, only `ANALYZE temp` here — this single-file
    // engine has no temp schema) analyzes nothing AND clears nothing, so it must not even
    // CREATE `sqlite_stat1`: real SQLite would target the temp schema and leave main's
    // `sqlite_master` untouched. Whole-db `ANALYZE`/`ANALYZE main` on an EMPTY database is
    // `DeleteScope::All` (not `None`), so it still creates the empty table — the documented
    // requirement — and every other scope has rows to clear or replace.
    if matches!(plan.delete, DeleteScope::None) {
        debug_assert!(plan.jobs.is_empty(), "a no-op (None) scope carries no measurement jobs");
        return Ok(());
    }
    let stat1_root = cat.ensure_sqlite_stat1(pager)?;
    let enc = text_encoding_of(&*pager);
    let rows = compute_rows(&*pager, &plan.jobs, enc)?;
    replace_stat_rows(pager, stat1_root, &plan.delete, &rows, enc)
}

// ---------------------------------------------------------------------------
// Scope resolution (pure over the catalog).
// ---------------------------------------------------------------------------

/// A resolved `ANALYZE`: which existing `sqlite_stat1` rows to clear, and which
/// b-trees to (re)measure.
struct Plan {
    delete: DeleteScope,
    jobs: Vec<Job>,
}

/// Which existing `sqlite_stat1` rows a run replaces. Mirrors SQLite's
/// `openStatTable`: a whole-database analyze clears everything; `ANALYZE <table>`
/// clears that table's rows (`WHERE tbl = name`); `ANALYZE <index>` clears just
/// that index's row (`WHERE idx = name`); a no-op scope clears nothing.
enum DeleteScope {
    All,
    ByTable(String),
    ByIndex(String),
    None,
}

/// One unit of measurement producing at most one `sqlite_stat1` row.
enum Job {
    /// Stream this index b-tree; emit `tbl | idx | "N a1 ... aK"` (K = `key_cols`).
    Index { tbl: String, idx: String, root: PageId, key_cols: usize },
    /// Count this rowid table's rows; emit `tbl | NULL | "N"` (a table with no
    /// index), so the planner still learns its size.
    TableCount { tbl: String, root: PageId },
}

/// Resolve `target` to a [`Plan`] against the catalog, erroring on an unknown name
/// BEFORE any write. Index names are checked before table names (SQLite's
/// `FindIndex` precedes `LocateTable`); a lone unqualified database name selects a
/// whole schema, taking precedence over a same-named table (SQLite resolves a bare
/// token as a database first).
fn resolve(cat: &SchemaCatalog, target: Option<&QualifiedName>) -> Result<Plan> {
    let Some(qn) = target else {
        return Ok(Plan { delete: DeleteScope::All, jobs: all_table_jobs(cat)? });
    };

    if qn.schema.is_none() {
        // `ANALYZE main` = analyze the whole main schema.
        if qn.name.eq_ignore_ascii_case("main") {
            return Ok(Plan { delete: DeleteScope::All, jobs: all_table_jobs(cat)? });
        }
        // `ANALYZE temp`: this single-file engine has no temp schema, so there is
        // nothing to analyze and nothing of main's to clear — a safe no-op.
        if qn.name.eq_ignore_ascii_case("temp") {
            return Ok(Plan { delete: DeleteScope::None, jobs: Vec::new() });
        }
    }

    // A schema qualifier (`main.t`) is otherwise ignored: this engine has one
    // schema, so the object is resolved by name within it.
    let name = &qn.name;
    if let Some(ix) = cat.index(name)? {
        return Ok(Plan { delete: DeleteScope::ByIndex(ix.name.clone()), jobs: index_jobs_for(cat, ix)? });
    }
    if let Some(tab) = cat.table(name)? {
        return Ok(Plan { delete: DeleteScope::ByTable(tab.name.clone()), jobs: table_jobs(cat, tab)? });
    }
    Err(Error::sql(format!("no such table: {name}")))
}

/// Jobs for every user table (`ANALYZE` with no target / `ANALYZE main`).
fn all_table_jobs(cat: &SchemaCatalog) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    // `tables()` already excludes the internal `sqlite_*` tables (`sqlite_schema`,
    // `sqlite_sequence`, `sqlite_stat1`), so ANALYZE never analyzes its own stat
    // table or the schema.
    for tab in cat.tables()? {
        table_jobs_into(cat, tab, &mut jobs)?;
    }
    Ok(jobs)
}

/// Jobs for one table and all its indexes (`ANALYZE <table>`).
fn table_jobs(cat: &SchemaCatalog, tab: &TableDef) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    table_jobs_into(cat, tab, &mut jobs)?;
    Ok(jobs)
}

fn table_jobs_into(cat: &SchemaCatalog, tab: &TableDef, jobs: &mut Vec<Job>) -> Result<()> {
    // SQLite's analyzeOneTable skips a `sqlite_*` system table (no user stats).
    // `tables()` already filters them from the whole-db scan, but an explicit
    // `ANALYZE sqlite_schema` reaches here, so guard again.
    if is_reserved_schema_name(&tab.name) {
        return Ok(());
    }

    // `indexes_on` returns the SECONDARY indexes only — the explicit `CREATE INDEX`es
    // plus the row-owning (`emit_row`) UNIQUE / PRIMARY KEY auto-indexes. It never
    // returns a WITHOUT ROWID table's PK (its `emit_row` is false: the table's own
    // b-tree IS the PK), which is exactly why that row is handled explicitly below.
    let indexes = cat.indexes_on(&tab.name)?;

    if tab.without_rowid {
        // A WITHOUT ROWID table stores its rows in its PRIMARY KEY b-tree
        // (`tab.root_page` IS that index). fileformat2 §2.6.4: its stat row has
        // `idx == tbl` and is the ordinary index stat over that PK b-tree, with `K`
        // = the PRIMARY KEY column count. The decoded key record stores the PK columns
        // FIRST (see `minisqlite-exec`'s `without_rowid::WrLayout`), so scanning the
        // first `K` columns is the PK prefix; the PK is unique, so its last integer is 1.
        if let Some(key_cols) = wr_pk_key_col_count(tab) {
            jobs.push(Job::Index {
                tbl: tab.name.clone(),
                idx: tab.name.clone(),
                root: tab.root_page,
                key_cols,
            });
        }
        // else: a genuinely PK-less (corrupt) catalog — impossible for a well-formed WR
        // table, which always declares a PRIMARY KEY (`without_rowid::wr_layout` fails
        // closed on the same emptiness). Skip rather than scan a phantom key.
    } else if indexes.is_empty() {
        // A ROWID table with no index gets the table-level `idx = NULL` row (stat =
        // row count), so the planner still learns its size (fileformat2 §2.6.4). A WR
        // table never lands here: its PK row above already carries the row count.
        jobs.push(Job::TableCount { tbl: tab.name.clone(), root: tab.root_page });
    }

    // Secondary-index rows, for a rowid OR a WITHOUT ROWID table. (CREATE INDEX on a WR
    // base table is rejected by this engine today, so `indexes` is empty for a WR table
    // in practice; the loop stays general so a future WR secondary index is analyzed.)
    for ix in indexes {
        push_index_job(tab, ix, jobs);
    }
    Ok(())
}

/// Jobs for a single named index (`ANALYZE <index>`) — that index only.
fn index_jobs_for(cat: &SchemaCatalog, ix: &IndexDef) -> Result<Vec<Job>> {
    // An index on a `sqlite_*` table carries no user stats (matches analyzeOneTable).
    if is_reserved_schema_name(&ix.table) {
        return Ok(Vec::new());
    }
    let Some(tab) = cat.table(&ix.table)? else {
        // Every index names an existing table; a dangling reference is corrupt schema.
        // Fail closed rather than fabricate a row for a table that is not there.
        return Err(Error::sql(format!("no such table: {}", ix.table)));
    };
    let mut jobs = Vec::new();
    push_index_job(tab, ix, &mut jobs);
    Ok(jobs)
}

fn push_index_job(tab: &TableDef, ix: &IndexDef, jobs: &mut Vec<Job>) {
    jobs.push(Job::Index {
        tbl: tab.name.clone(),
        idx: ix.name.clone(),
        root: ix.root_page,
        // The b-tree stores one key column per `columns` entry (an expression index
        // stores its computed values in the same slots), so `columns.len()` is the K
        // whose K+1 stat integers the scan produces.
        key_cols: ix.columns.len(),
    });
}

/// The number of PRIMARY KEY key-columns of a WITHOUT ROWID table — the width `K` of
/// the distinct-prefix scan over its PK b-tree (`tab.root_page` IS that index).
///
/// Read from the single authority [`TableDef::primary_key`] (the ordered PK-column
/// indices, populated for EVERY PK form), the same source `minisqlite-exec`'s
/// `without_rowid::wr_pk_columns` + `wr_layout` read to build the stored key — so this
/// `K` equals that key's width by construction. It recovers the shape the older spec+flag
/// derivation missed: a *table-level single INTEGER PRIMARY KEY* (`PRIMARY KEY(a)` with
/// `a INTEGER`, WITHOUT ROWID), which reserves no `emit_row == false` auto-index ordinal
/// (an integer PK is excluded) AND sets no per-column flag (a table-level constraint) —
/// closing the latent defect for the WR PK stat row (the same `TableDef::primary_key` fix
/// applied in FK resolution and `wr_pk_columns`).
///
/// Counts the DISTINCT PK columns: `wr_layout` de-duplicates a repeated PK member when it
/// builds the key prefix (`pk_len`), and this `K` must equal that width, so the de-dup is
/// mirrored here. `None` only for a genuinely PK-less (corrupt) catalog — impossible for a
/// well-formed WR table, on which `wr_layout` also fails closed on the same emptiness.
fn wr_pk_key_col_count(tab: &TableDef) -> Option<usize> {
    let mut idxs: Vec<usize> = Vec::new();
    for &i in &tab.primary_key {
        if !idxs.contains(&i) {
            idxs.push(i);
        }
    }
    (!idxs.is_empty()).then_some(idxs.len())
}

// ---------------------------------------------------------------------------
// Measurement (streaming reads over the b-trees).
// ---------------------------------------------------------------------------

/// One computed `sqlite_stat1` row.
struct StatRow {
    tbl: String,
    /// `Some(index_name)` for an index row, `None` for a table-level `idx = NULL` row.
    idx: Option<String>,
    stat: String,
}

fn compute_rows(pager: &dyn Pager, jobs: &[Job], enc: TextEncoding) -> Result<Vec<StatRow>> {
    let mut out = Vec::with_capacity(jobs.len());
    for job in jobs {
        match job {
            Job::Index { tbl, idx, root, key_cols } => {
                // Skip an empty index (no rows to describe), matching SQLite.
                if let Some(stat) = scan_index(pager, *root, *key_cols, enc)? {
                    out.push(StatRow { tbl: tbl.clone(), idx: Some(idx.clone()), stat });
                }
            }
            Job::TableCount { tbl, root } => {
                let n = count_table_rows(pager, *root)?;
                if n > 0 {
                    out.push(StatRow { tbl: tbl.clone(), idx: None, stat: n.to_string() });
                }
            }
        }
    }
    Ok(out)
}

/// Stream one index b-tree in key order and build its `stat` string, or `None` if
/// the index is empty. O(`key_cols`) memory: only two reused decoded-key buffers and the
/// per-prefix change counters are held — the index is never collected into a `Vec`.
fn scan_index(
    pager: &dyn Pager,
    root: PageId,
    key_cols: usize,
    enc: TextEncoding,
) -> Result<Option<String>> {
    debug_assert!(key_cols >= 1, "an index has at least one key column");
    let mut cursor = IndexCursor::open(pager, root)?;
    let mut n_row: u64 = 0;
    // `changes[i]` = number of adjacent key pairs that differ within their first
    // `i + 1` columns, so the distinct count over those columns is `changes[i] + 1`.
    let mut changes = vec![0u64; key_cols];
    // Two buffers swapped each row: `prev` holds the last decoded key, `cur` is the one
    // decoded into (cleared first by `decode_record_into_enc`). Reusing them keeps hot-loop
    // allocation bounded — no fresh `Vec<Value>` spine per row (only TEXT/BLOB column
    // bodies still allocate) — the bounded per-row allocation discipline. `have_prev`
    // gates the first row, which has no predecessor to compare.
    let mut prev: Vec<Value> = Vec::new();
    let mut cur: Vec<Value> = Vec::new();
    let mut have_prev = false;

    let mut positioned = cursor.first()?;
    while positioned {
        n_row += 1;
        let key = cursor.key()?;
        decode_record_into_enc(&key, enc, &mut cur);
        // `key_cols` must never exceed the stored key width, or the scan would read past
        // the key into `first_diff_col`'s NULL padding and emit a too-wide stat. An index
        // key is K columns + rowid; a WITHOUT ROWID PK b-tree is the PK columns + the rest
        // — both are >= key_cols. If a future WR PK-derivation drift (see
        // `wr_pk_key_col_count`) ever over-counts, this trips in debug rather than writing
        // a corrupt row.
        debug_assert!(
            key_cols <= cur.len(),
            "stat key_cols {key_cols} exceeds decoded key width {}",
            cur.len()
        );
        if have_prev {
            // When two adjacent keys first differ at column `j`, every prefix of
            // length `>= j + 1` changes at this boundary, so bump `changes[j..]`.
            if let Some(j) = first_diff_col(&prev, &cur, key_cols) {
                for c in &mut changes[j..] {
                    *c += 1;
                }
            }
        }
        // `prev` now holds this row; the old `prev` buffer becomes `cur` for reuse (its
        // stale contents are cleared by the next `decode_record_into_enc`).
        std::mem::swap(&mut prev, &mut cur);
        have_prev = true;
        positioned = cursor.next()?;
    }

    if n_row == 0 {
        Ok(None)
    } else {
        Ok(Some(index_stat_text(n_row, &changes)))
    }
}

/// Count rows in a rowid table by streaming it in rowid order (O(1) memory).
fn count_table_rows(pager: &dyn Pager, root: PageId) -> Result<u64> {
    let mut cursor = TableCursor::open(pager, root)?;
    let mut n: u64 = 0;
    let mut positioned = cursor.first()?;
    while positioned {
        n += 1;
        positioned = cursor.next()?;
    }
    Ok(n)
}

/// The left-most of the first `k` columns where `prev` and `cur` differ (so every
/// prefix of that length or more is a new distinct value), or `None` if all `k` are
/// equal. A key shorter than `k` reads a missing column as `NULL` (the record decoder's
/// short-row rule; defensive — a real index key always has all `k` columns plus a
/// trailing rowid).
///
/// A NULL is treated as a value DISTINCT from every other, INCLUDING another NULL, so a
/// column counts as "differs" whenever either side is NULL. This is required by
/// `fileformat2` §2.6.4: a UNIQUE index legitimately holds multiple NULLs, yet its last
/// stat integer must be 1 unconditionally — which only holds if each NULL is its own
/// distinct prefix (otherwise a run of `d` duplicate NULLs would collapse the full-key
/// distinct count below `N` and push the last integer above 1). Every NON-NULL pair
/// still uses the ordinary [`Collation::Binary`] comparison (the b-tree's own order), so
/// non-NULL statistics are unchanged; only NULLs flip from "equal" to "distinct" here.
fn first_diff_col(prev: &[Value], cur: &[Value], k: usize) -> Option<usize> {
    for i in 0..k {
        let a = prev.get(i).unwrap_or(&Value::Null);
        let b = cur.get(i).unwrap_or(&Value::Null);
        let differs = matches!(a, Value::Null)
            || matches!(b, Value::Null)
            || compare_values(a, b, Collation::Binary) != Ordering::Equal;
        if differs {
            return Some(i);
        }
    }
    None
}

/// Build `"N a1 a2 ... aK"` from the row count and the per-prefix change counts.
fn index_stat_text(n_row: u64, changes: &[u64]) -> String {
    let mut s = n_row.to_string();
    for &c in changes {
        s.push(' ');
        s.push_str(&avg_rows_per_distinct(n_row, c + 1).to_string());
    }
    s
}

/// The per-prefix stat integer: `round(n_row / n_distinct)`, computed as
/// `(n_row + n_distinct / 2) / n_distinct`. A spec-correct APPROXIMATION of §2.6.4's
/// "estimated average number of rows" (the docs give no exact formula and call it
/// "approximate"), NOT a reproduction of any real-SQLite internal. With
/// `n_distinct == n_row` (a unique prefix) it is exactly 1 — the spec's one stated
/// invariant for a unique index's last integer.
fn avg_rows_per_distinct(n_row: u64, n_distinct: u64) -> u64 {
    debug_assert!(n_distinct >= 1, "a non-empty index has >= 1 distinct prefix value");
    debug_assert!(n_distinct <= n_row, "distinct prefixes cannot exceed the row count");
    (n_row + n_distinct / 2) / n_distinct
}

// ---------------------------------------------------------------------------
// Replace the in-scope rows (delete, then insert fresh).
// ---------------------------------------------------------------------------

fn replace_stat_rows(
    pager: &mut dyn Pager,
    stat1_root: PageId,
    scope: &DeleteScope,
    rows: &[StatRow],
    enc: TextEncoding,
) -> Result<()> {
    // Collect the doomed rowids first, THEN delete: mutating the b-tree while a read
    // cursor is open over it would invalidate the cursor.
    let doomed = stat1_rowids_in_scope(&*pager, stat1_root, scope, enc)?;
    for rowid in doomed {
        table_delete(pager, stat1_root, rowid)?;
    }

    // Fresh rows go in at rowids past the current max, so they never collide with a
    // surviving row (a partial `ByTable`/`ByIndex` clear leaves others in place).
    let mut next_rowid = next_stat1_rowid(&*pager, stat1_root)?;
    // Write `sqlite_stat1` rows in the database's text encoding (`enc`), matching the
    // delete pass above (which decoded with the same `enc`) and every other data-row /
    // schema-row writer (`SchemaRow::to_record_enc`, INSERT, index maintenance). On a
    // UTF-8 database this is byte-identical to the plain codec; on a UTF-16 database the
    // TEXT columns are stored in that encoding so real sqlite reads the stats back and a
    // re-ANALYZE clears its own stale rows.
    for row in rows {
        let idx = match &row.idx {
            Some(name) => Value::Text(name.clone()),
            None => Value::Null,
        };
        let record = encode_record_enc(
            &[Value::Text(row.tbl.clone()), idx, Value::Text(row.stat.clone())],
            enc,
        );
        table_insert(pager, stat1_root, next_rowid, &record)?;
        next_rowid += 1;
    }
    Ok(())
}

/// The rowids of `sqlite_stat1` rows in `scope`, gathered by one streaming pass.
fn stat1_rowids_in_scope(
    pager: &dyn Pager,
    stat1_root: PageId,
    scope: &DeleteScope,
    enc: TextEncoding,
) -> Result<Vec<i64>> {
    if matches!(scope, DeleteScope::None) {
        return Ok(Vec::new());
    }
    let mut cursor = TableCursor::open(pager, stat1_root)?;
    let mut out = Vec::new();
    let mut positioned = cursor.first()?;
    while positioned {
        let hit = {
            let payload = cursor.payload()?;
            in_scope(scope, &decode_record_enc(&payload, enc))
        };
        if hit {
            out.push(cursor.rowid());
        }
        positioned = cursor.next()?;
    }
    Ok(out)
}

/// Whether a decoded `sqlite_stat1` record `(tbl, idx, stat)` is in the delete
/// scope. `tbl` is column 0 and `idx` column 1; the comparison is BINARY (the
/// column has no declared collation), matching the exact-name identity we wrote.
fn in_scope(scope: &DeleteScope, record: &[Value]) -> bool {
    match scope {
        DeleteScope::None => false,
        DeleteScope::All => true,
        DeleteScope::ByTable(name) => matches_text(record.first(), name),
        DeleteScope::ByIndex(name) => matches_text(record.get(1), name),
    }
}

fn matches_text(cell: Option<&Value>, name: &str) -> bool {
    matches!(cell, Some(Value::Text(s)) if s == name)
}

/// The next free rowid for `sqlite_stat1` (largest present + 1, or 1 when empty).
fn next_stat1_rowid(pager: &dyn Pager, stat1_root: PageId) -> Result<i64> {
    let mut cursor = TableCursor::open(pager, stat1_root)?;
    Ok(if cursor.last()? { cursor.rowid() + 1 } else { 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_is_round_to_nearest() {
        // (n_row + n_distinct / 2) / n_distinct — round-to-nearest of N / nDistinct.
        assert_eq!(avg_rows_per_distinct(6, 3), 2); // 6/3 = 2
        assert_eq!(avg_rows_per_distinct(5, 2), 3); // 5/2 = 2.5 -> 3
        assert_eq!(avg_rows_per_distinct(5, 4), 1); // 5/4 = 1.25 -> 1
        assert_eq!(avg_rows_per_distinct(7, 4), 2); // 7/4 = 1.75 -> 2
        assert_eq!(avg_rows_per_distinct(10, 3), 3); // 10/3 = 3.33 -> 3
    }

    #[test]
    fn avg_of_a_unique_prefix_is_one() {
        // A unique K-th column has n_distinct == n_row, so the last integer is 1 —
        // the one invariant fileformat2 §2.6.4 states. Exhaust a wide range.
        for n in 1..=1000u64 {
            assert_eq!(avg_rows_per_distinct(n, n), 1, "unique last integer must be 1 (n={n})");
        }
    }

    #[test]
    fn stat_text_single_column() {
        // 6 rows, 3 distinct values in column 0 (changes[0] = 2): "6 2".
        assert_eq!(index_stat_text(6, &[2]), "6 2");
    }

    #[test]
    fn stat_text_two_columns_rounds_each_prefix() {
        // 5 rows; distinct(col0) = 2 (changes[0]=1) -> 5/2 -> 3; distinct(col0,col1)
        // = 4 (changes[1]=3) -> 5/4 -> 1: "5 3 1".
        assert_eq!(index_stat_text(5, &[1, 3]), "5 3 1");
    }

    #[test]
    fn stat_text_unique_multicolumn_ends_in_one() {
        // 4 rows, 2 distinct in col0, all 4 distinct on the full key: "4 2 1".
        assert_eq!(index_stat_text(4, &[1, 3]), "4 2 1");
    }

    #[test]
    fn stat_text_all_equal_single_group() {
        // 4 identical rows: one distinct value, so the average group size is 4: "4 4".
        assert_eq!(index_stat_text(4, &[0]), "4 4");
    }

    #[test]
    fn first_diff_col_finds_leftmost_change() {
        let a = [Value::Integer(1), Value::Integer(2), Value::Integer(9)];
        let b = [Value::Integer(1), Value::Integer(5), Value::Integer(9)];
        assert_eq!(first_diff_col(&a, &b, 3), Some(1));
        assert_eq!(first_diff_col(&a, &b, 1), None); // first column equal
        assert_eq!(first_diff_col(&a, &a, 3), None); // identical
    }

    #[test]
    fn first_diff_col_reads_a_missing_tail_as_a_distinct_null() {
        let a = [Value::Integer(1)];
        let b = [Value::Integer(1)];
        // k=1 looks only at column 0, both the non-NULL 1 → equal.
        assert_eq!(first_diff_col(&a, &b, 1), None);
        // Both short at column 1, read as NULL; NULL is distinct from everything
        // (including another NULL), so they differ there. (Defensive: a real index key
        // always carries all k columns plus the trailing rowid.)
        assert_eq!(first_diff_col(&a, &b, 2), Some(1));
        // A present value also differs from the implicit NULL of a shorter key.
        let c = [Value::Integer(1), Value::Integer(7)];
        assert_eq!(first_diff_col(&a, &c, 2), Some(1));
    }

    #[test]
    fn leading_nulls_are_each_a_distinct_prefix() {
        // fileformat2 §2.6.4: a UNIQUE index's last stat integer is 1 unconditionally,
        // even over duplicate NULLs — so each NULL is its OWN distinct value, INCLUDING
        // vs another NULL. Two NULL-leading keys therefore differ at column 0.
        let a = [Value::Null, Value::Integer(1)];
        let b = [Value::Null, Value::Integer(2)];
        assert_eq!(first_diff_col(&a, &b, 1), Some(0), "NULL != NULL: distinct at col 0");
        assert_eq!(first_diff_col(&a, &b, 2), Some(0), "leftmost difference is still col 0");
        // A NULL differs from a non-NULL as well.
        let c = [Value::Integer(1), Value::Integer(1)];
        assert_eq!(first_diff_col(&a, &c, 1), Some(0), "NULL vs 1 differs at col 0");
        // With no NULLs, ordinary Binary comparison is unchanged: equal col 0 defers to
        // the next column.
        let d = [Value::Integer(5), Value::Integer(1)];
        let e = [Value::Integer(5), Value::Integer(2)];
        assert_eq!(first_diff_col(&d, &e, 2), Some(1), "non-NULL pairs unaffected");
    }

    #[test]
    fn text_match_is_binary_case_sensitive() {
        let rec = [Value::Text("t".into()), Value::Null, Value::Text("2".into())];
        assert!(in_scope(&DeleteScope::ByTable("t".into()), &rec));
        assert!(!in_scope(&DeleteScope::ByTable("T".into()), &rec)); // BINARY: case-sensitive
        assert!(in_scope(&DeleteScope::All, &rec));
        assert!(!in_scope(&DeleteScope::None, &rec));
        // idx is NULL here, so an idx-scoped delete never matches.
        assert!(!in_scope(&DeleteScope::ByIndex("x".into()), &rec));
    }

    // -- WITHOUT ROWID PK key-column derivation (drives the idx == tbl stat row) -------

    /// Build a `TableDef` through the real catalog create path, so `auto_indexes` /
    /// `primary_key` flags are populated exactly as production does.
    fn wr_table_def(create_sql: &str) -> (minisqlite_pager::MemPager, SchemaCatalog, String) {
        use minisqlite_btree::init_database;
        use minisqlite_pager::MemPager;
        use minisqlite_sql::{parse, Statement};
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let mut cat = SchemaCatalog::new();
        let ast = parse(create_sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else {
            panic!("not a CREATE TABLE: {create_sql}");
        };
        let name = stmt.name.name.clone();
        cat.create_table(&mut pager, stmt, create_sql).unwrap();
        (pager, cat, name)
    }

    fn pk_cols(cat: &SchemaCatalog, name: &str) -> Option<usize> {
        wr_pk_key_col_count(cat.table(name).unwrap().unwrap())
    }

    #[test]
    fn wr_pk_count_single_column_pk_is_one() {
        let (_p, cat, t) = wr_table_def("CREATE TABLE t(a, b, PRIMARY KEY(a)) WITHOUT ROWID");
        assert_eq!(pk_cols(&cat, &t), Some(1), "one PK column → K = 1");
    }

    #[test]
    fn wr_pk_count_composite_pk_is_two() {
        let (_p, cat, t) =
            wr_table_def("CREATE TABLE t(a, b, c, PRIMARY KEY(c, a)) WITHOUT ROWID");
        assert_eq!(pk_cols(&cat, &t), Some(2), "two PK columns → K = 2");
    }

    #[test]
    fn wr_pk_count_column_level_integer_pk_is_one() {
        // A column-level INTEGER PRIMARY KEY reserves no auto-index spec (an integer PK is
        // excluded); it is recovered from `TableDef::primary_key`, the single ordered
        // authority for every PK form (no longer the per-column flag fallback) → K = 1.
        let (_p, cat, t) = wr_table_def("CREATE TABLE t(a INTEGER PRIMARY KEY, b) WITHOUT ROWID");
        assert_eq!(pk_cols(&cat, &t), Some(1), "INTEGER PK recovered via TableDef::primary_key");
    }

    #[test]
    fn wr_pk_count_table_level_integer_pk_is_one() {
        // The shape that WAS the "unrecoverable residual": a table-level single INTEGER
        // PRIMARY KEY reserves no auto-index ordinal AND sets no per-column flag, yet
        // `TableDef::primary_key` records it (== `[0]`), so K = 1 is recovered and the WR
        // PK stat row is emitted (mirrors `without_rowid::wr_pk_columns`) — now CLOSED.
        let (_p, cat, t) =
            wr_table_def("CREATE TABLE t(a INTEGER, b, PRIMARY KEY(a)) WITHOUT ROWID");
        assert_eq!(pk_cols(&cat, &t), Some(1), "table-level INTEGER PK recovered → K = 1");
    }
}
