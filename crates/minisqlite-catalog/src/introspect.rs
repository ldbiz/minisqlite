//! Shared schema-introspection row building for the `PRAGMA` statement forms
//! (`table_info`, `table_xinfo`, `index_list`, `index_info`, `index_xinfo`,
//! `foreign_key_list`) AND their `pragma_*` table-valued-function equivalents
//! (pragma.html §2 "PRAGMA functions").
//!
//! Both surfaces read the SAME [`Catalog`] and must produce byte-identical columns and
//! rows, so the row-building lives HERE, once, over the `Catalog` seam — the engine's
//! statement path and the planner/executor's TVF path both call in. There is deliberately
//! no second copy to drift: `SELECT * FROM pragma_table_info('t')` and
//! `PRAGMA table_info(t)` are the same rows because they run the same code.
//!
//! Every builder is a pure, O(#columns)/O(#indexes) read over the cached schema — no
//! data-page access — and resolves its target namespace the way SQLite name resolution
//! does: a schema qualifier (`main`/`temp`/an attach alias) via
//! [`Catalog::db_of_schema`], else an unqualified object name via
//! [`Catalog::owner_db`] (a TABLE argument) or [`Catalog::owner_db_of_index`] (an INDEX
//! argument). A `None` name, an unknown qualifier, or an absent object each yields zero
//! rows (the fixed columns still stand), which is SQLite's convention for these pragmas.
//!
//! `index_info` / `index_xinfo` take an INDEX argument but fall back to a TABLE one: when the
//! name resolves to no index they retry it as a WITHOUT ROWID table (pragma.html, since SQLite
//! 3.30.0) via [`Catalog::owner_db`], reporting that table's underlying b-tree columns.

use crate::catalog::Catalog;
use crate::def::{ColumnDef, IndexDef, ReferentialAction, TableDef};
use minisqlite_types::{DbIndex, Result, Value};

/// Which schema-introspection PRAGMA a `pragma_*` table-valued function reflects. Each maps
/// to one builder with the same columns/rows as the corresponding `PRAGMA` statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PragmaFunction {
    /// `pragma_table_info(T)` — one row per NORMAL column.
    TableInfo,
    /// `pragma_table_xinfo(T)` — one row per column incl. generated, plus `hidden`.
    TableXInfo,
    /// `pragma_index_info(IDX)` — one row per key column of an index.
    IndexInfo,
    /// `pragma_index_xinfo(IDX)` — one row per index-record column (key + auxiliary).
    IndexXInfo,
    /// `pragma_index_list(T)` — one row per index on a table.
    IndexList,
    /// `pragma_foreign_key_list(T)` — one row per column of each FK on a table.
    ForeignKeyList,
}

impl PragmaFunction {
    /// Classify a FROM-clause function name (case-insensitive) as one of the introspection
    /// pragma table-valued functions, or `None` for any other name. The name is the full
    /// `pragma_`-prefixed spelling the TVF form uses (`pragma_table_info`); the statement
    /// form (`table_info`) is dispatched by the engine directly, not through here.
    pub fn from_tvf_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pragma_table_info" => Some(Self::TableInfo),
            "pragma_table_xinfo" => Some(Self::TableXInfo),
            "pragma_index_info" => Some(Self::IndexInfo),
            "pragma_index_xinfo" => Some(Self::IndexXInfo),
            "pragma_index_list" => Some(Self::IndexList),
            "pragma_foreign_key_list" => Some(Self::ForeignKeyList),
            _ => None,
        }
    }

    /// The fixed result column names in order — identical to the corresponding PRAGMA
    /// statement's columns (pragma.html §2). The TVF's derived-source schema and the
    /// statement's `QueryResult.columns` both come from here so they cannot diverge.
    pub fn columns(self) -> Vec<String> {
        self.column_names().iter().map(|s| s.to_string()).collect()
    }

    /// The static column-name slice backing [`columns`](Self::columns).
    pub fn column_names(self) -> &'static [&'static str] {
        match self {
            Self::TableInfo => &["cid", "name", "type", "notnull", "dflt_value", "pk"],
            Self::TableXInfo => &["cid", "name", "type", "notnull", "dflt_value", "pk", "hidden"],
            Self::IndexInfo => &["seqno", "cid", "name"],
            Self::IndexXInfo => &["seqno", "cid", "name", "desc", "coll", "key"],
            Self::IndexList => &["seq", "name", "unique", "origin", "partial"],
            Self::ForeignKeyList => {
                &["id", "seq", "table", "from", "to", "on_update", "on_delete", "match"]
            }
        }
    }
}

/// Build the rows for one pragma introspection function, resolved against the live schema
/// `cat`. `schema` is the optional schema qualifier and `name` the object argument (a
/// TABLE name, or an INDEX name for `index_info`/`index_xinfo`). See the module docs for
/// the empty-result cases. The returned rows are exactly `self.columns()` wide.
pub fn pragma_rows(
    cat: &dyn Catalog,
    kind: PragmaFunction,
    schema: Option<&str>,
    name: Option<&str>,
) -> Result<Vec<Vec<Value>>> {
    let Some(name) = name else {
        return Ok(Vec::new());
    };
    match kind {
        PragmaFunction::TableInfo => table_info_rows(cat, schema, name),
        PragmaFunction::TableXInfo => table_xinfo_rows(cat, schema, name),
        PragmaFunction::IndexList => index_list_rows(cat, schema, name),
        PragmaFunction::IndexInfo => index_info_rows(cat, schema, name),
        PragmaFunction::IndexXInfo => index_xinfo_rows(cat, schema, name),
        PragmaFunction::ForeignKeyList => foreign_key_list_rows(cat, schema, name),
    }
}

/// The target namespace for a TABLE-argument pragma: the schema qualifier if given
/// (unknown → `None`), else the unqualified name's owner in SQLite search order (temp,
/// main, attached). This is the SAME rule `SELECT ... FROM t` uses, so a pragma never
/// disagrees with SQL on which object an unqualified name denotes.
fn resolve_table_target(
    cat: &dyn Catalog,
    schema: Option<&str>,
    name: &str,
) -> Result<Option<DbIndex>> {
    match schema {
        Some(schema) => cat.db_of_schema(schema),
        None => cat.owner_db(name),
    }
}

/// The target namespace for an INDEX-argument pragma (`index_info`, `index_xinfo`): the
/// schema qualifier if given, else the unqualified INDEX name's owner in search order (via
/// [`Catalog::owner_db_of_index`]). Indexes have their own search — an index name is not a
/// table/view name — so this cannot reuse [`resolve_table_target`].
fn resolve_index_target(
    cat: &dyn Catalog,
    schema: Option<&str>,
    name: &str,
) -> Result<Option<DbIndex>> {
    match schema {
        Some(schema) => cat.db_of_schema(schema),
        None => cat.owner_db_of_index(name),
    }
}

/// `table_info(T)` rows — one per NORMAL column: `cid`, `name`, `type` (declared type
/// verbatim, else `''`), `notnull` (0/1), `dflt_value` (default text or NULL), `pk` (0, or
/// the 1-based position within the primary key).
///
/// GENERATED columns are OMITTED (pragma.html: "does not show information about generated
/// columns … Use PRAGMA table_xinfo"). `cid` is therefore the rank within the RESULT SET —
/// emitted columns numbered 0,1,2,… with no gap where a generated column was skipped
/// (matching real sqlite's `i - nHidden`), while `pk` is computed from the column's TRUE
/// table index.
fn table_info_rows(cat: &dyn Catalog, schema: Option<&str>, name: &str) -> Result<Vec<Vec<Value>>> {
    let mut rows = Vec::new();
    if let Some(db) = resolve_table_target(cat, schema, name)? {
        if let Some(tbl) = cat.table_in(db, name)? {
            let mut cid: i64 = 0;
            for (i, col) in tbl.columns.iter().enumerate() {
                if col.generated.is_some() {
                    continue;
                }
                rows.push(table_info_cells(cid, col, tbl, i));
                cid += 1;
            }
        }
    }
    Ok(rows)
}

/// `table_xinfo(T)` rows — like [`table_info_rows`] but one row for EVERY column (incl.
/// generated) with a trailing `hidden` flag (0 normal, 2 VIRTUAL, 3 STORED generated).
/// Because every column is listed, `cid` is the column's TRUE 0-based table index (no
/// renumbering); the rows whose `hidden` is non-zero are exactly the ones `table_info`
/// omits.
fn table_xinfo_rows(cat: &dyn Catalog, schema: Option<&str>, name: &str) -> Result<Vec<Vec<Value>>> {
    let mut rows = Vec::new();
    if let Some(db) = resolve_table_target(cat, schema, name)? {
        if let Some(tbl) = cat.table_in(db, name)? {
            for (i, col) in tbl.columns.iter().enumerate() {
                let mut cells = table_info_cells(i as i64, col, tbl, i);
                cells.push(Value::Integer(column_hidden_flag(col)));
                rows.push(cells);
            }
        }
    }
    Ok(rows)
}

/// `index_list(T)` rows — one per index on `T`: `seq` (0-based creation-order rank),
/// `name`, `unique` (0/1), `origin` (`'c'`/`'u'`/`'pk'`), `partial` (0/1). Indexes are
/// taken in the catalog's creation order.
fn index_list_rows(cat: &dyn Catalog, schema: Option<&str>, name: &str) -> Result<Vec<Vec<Value>>> {
    let mut rows = Vec::new();
    if let Some(db) = resolve_table_target(cat, schema, name)? {
        let tbl = cat.table_in(db, name)?;
        for (seq, idx) in cat.indexes_on_in(db, name)?.iter().enumerate() {
            rows.push(vec![
                Value::Integer(seq as i64),
                Value::Text(idx.name.clone()),
                Value::Integer(i64::from(idx.unique)),
                Value::Text(index_origin(idx, tbl).to_string()),
                Value::Integer(i64::from(idx.partial)),
            ]);
        }
    }
    Ok(rows)
}

/// `index_info(NAME)` rows. NAME is normally an INDEX: one row per KEY column — `seqno`
/// (0-based rank within the index), `cid` (the column's ordinal in the indexed table, -1 when
/// unresolvable, or -2 for an EXPRESSION key column), `name` (the column name, or NULL for an
/// expression key column). The `cid`/`name` rule for a key column lives in
/// [`index_column_cid_name`].
///
/// FALLBACK (pragma.html #pragma_index_info, since SQLite 3.30.0): when NO index of that name
/// exists but a WITHOUT ROWID TABLE does, the pragma instead returns that table's PRIMARY KEY
/// columns as they key the underlying b-tree — see [`index_info_rows_for_wr_table`]. The
/// table is resolved as a TABLE (its own search order / schema qualifier), tried only after
/// the index lookup misses because names share one namespace (a name is an index XOR a table).
/// A rowid table, an unknown name, or a WITHOUT ROWID table with no modelled PK all yield zero
/// rows.
fn index_info_rows(cat: &dyn Catalog, schema: Option<&str>, name: &str) -> Result<Vec<Vec<Value>>> {
    if let Some(db) = resolve_index_target(cat, schema, name)? {
        if let Some(idx) = cat.index_in(db, name)? {
            // The index and its table share a namespace, so read the table from the same
            // `db` (so `cid` resolves against the right columns).
            let tbl = cat.table_in(db, &idx.table)?;
            return Ok(index_info_rows_for(idx, tbl));
        }
    }
    if let Some(db) = resolve_table_target(cat, schema, name)? {
        if let Some(tbl) = cat.table_in(db, name)? {
            if tbl.without_rowid {
                return Ok(index_info_rows_for_wr_table(tbl));
            }
        }
    }
    Ok(Vec::new())
}

/// The de-duplicated PRIMARY KEY columns of `tbl`, in PRIMARY KEY order: [`primary_key_key_meta`]
/// with a repeated table-column index collapsed to its FIRST occurrence (keeping that
/// occurrence's rank / `desc` / `coll`). This is the record-level de-duplication both WITHOUT
/// ROWID pragmas require — pragma.html pins "duplicate columns removed" (index_info) and
/// "de-duplicated PRIMARY KEY columns" (index_xinfo) — so the pragma agrees with the on-disk
/// record, which stores each PK column once (the executor's `wr_layout` applies the same
/// `in_pk` collapse).
///
/// The collapse lives here rather than in [`primary_key_key_meta`] because that shared function
/// maps `TableDef.primary_key` 1:1 and the builder records a literal `PRIMARY KEY(a, a)` as
/// `[0, 0]` (duplicate PK columns are accepted at create time). The root fix is to de-duplicate
/// `TableDef.primary_key` in the builder, which would also correct the WITHOUT ROWID
/// secondary-index auxiliary path ([`index_xinfo_aux_rows`], which still reads the non-collapsed
/// [`primary_key_key_meta`]); both are out of this change's single-file scope.
fn dedup_pk_key_meta(tbl: &TableDef) -> Vec<PkKeyMeta> {
    let mut seen: Vec<usize> = Vec::new();
    primary_key_key_meta(tbl)
        .into_iter()
        .filter(|meta| {
            if seen.contains(&meta.col_idx) {
                false
            } else {
                seen.push(meta.col_idx);
                true
            }
        })
        .collect()
}

/// The `index_info` rows for a WITHOUT ROWID TABLE named as the pragma argument when no index
/// of that name exists (pragma.html #pragma_index_info, since SQLite 3.30.0): one row per
/// de-duplicated PRIMARY KEY column, in PRIMARY KEY order — the columns as they key the
/// underlying b-tree record. `seqno` is the 0-based rank within that key, `cid` the column's
/// TABLE ordinal (not its position in the reordered record), `name` the column name.
///
/// PK-column order/`desc`/`coll` come from [`primary_key_key_meta`]; a repeated column is
/// collapsed by [`dedup_pk_key_meta`] so a literal `PRIMARY KEY(a, a)` lists `a` once (the
/// b-tree stores it once). A WITHOUT ROWID table always has a PRIMARY KEY (create-time
/// validation), but a foreign/corrupt schema presenting an empty one (or a PK column index past
/// the column list) yields no such row rather than panicking.
fn index_info_rows_for_wr_table(tbl: &TableDef) -> Vec<Vec<Value>> {
    debug_assert!(tbl.without_rowid, "WR-table index_info fallback needs WITHOUT ROWID");
    dedup_pk_key_meta(tbl)
        .iter()
        .filter_map(|meta| tbl.columns.get(meta.col_idx).map(|col| (meta.col_idx, &col.name)))
        .enumerate()
        .map(|(seqno, (cid, name))| {
            vec![
                Value::Integer(seqno as i64),
                Value::Integer(cid as i64),
                Value::Text(name.clone()),
            ]
        })
        .collect()
}

/// The `index_info` rows for an already-resolved index `idx` and its (optional) table `tbl` —
/// the pure core of [`index_info_rows`], testable without a [`Catalog`]. One row per key
/// column: its `seqno`, then the `(cid, name)` pair from [`index_column_cid_name`].
fn index_info_rows_for(idx: &IndexDef, tbl: Option<&TableDef>) -> Vec<Vec<Value>> {
    idx.columns
        .iter()
        .enumerate()
        .map(|(seqno, colname)| {
            let (cid, name) = index_column_cid_name(idx, tbl, seqno, colname);
            vec![Value::Integer(seqno as i64), Value::Integer(cid), name]
        })
        .collect()
}

/// `index_xinfo(IDX)` rows — like [`index_info_rows`] but one row for EVERY column in the
/// index record (key columns first, then the auxiliary columns), with the extra
/// `desc`/`coll`/`key` columns (pragma.html #pragma_index_xinfo). The auxiliary columns are
/// the ones that locate the table row: a rowid table's single rowid (`cid` = -1, `name` =
/// NULL, `key` = 0), or a WITHOUT ROWID table's PRIMARY KEY columns that are not already
/// carried as key columns (see [`index_xinfo_aux_rows`]).
///
/// `desc` (reverse sort) and `coll` (collating sequence) are read per key column from the
/// parallel [`IndexDef::key_columns`]: `desc` is that column's `descending` flag, and
/// `coll` follows the collation inheritance chain (explicit per-key `COLLATE`, else the
/// table column's declared `COLLATE`, else literal `BINARY`), reported verbatim in the case
/// written.
///
/// FALLBACK (pragma.html #pragma_index_xinfo, since SQLite 3.30.0): when NO index of that name
/// exists but a WITHOUT ROWID TABLE does, the pragma instead returns that table's columns as
/// they appear in the underlying b-tree record — see [`index_xinfo_rows_for_wr_table`]. The
/// table is resolved as a TABLE, tried only after the index lookup misses (index XOR table
/// name). A rowid table, an unknown name, or a WITHOUT ROWID table with no modelled PK all
/// yield zero rows.
fn index_xinfo_rows(cat: &dyn Catalog, schema: Option<&str>, name: &str) -> Result<Vec<Vec<Value>>> {
    if let Some(db) = resolve_index_target(cat, schema, name)? {
        if let Some(idx) = cat.index_in(db, name)? {
            let tbl = cat.table_in(db, &idx.table)?;
            return Ok(index_xinfo_rows_for(idx, tbl));
        }
    }
    if let Some(db) = resolve_table_target(cat, schema, name)? {
        if let Some(tbl) = cat.table_in(db, name)? {
            if tbl.without_rowid {
                return Ok(index_xinfo_rows_for_wr_table(tbl));
            }
        }
    }
    Ok(Vec::new())
}

/// The `index_xinfo` rows for a WITHOUT ROWID TABLE named as the pragma argument when no index
/// of that name exists (pragma.html #pragma_index_xinfo, since SQLite 3.30.0): the table's
/// columns as they appear in the underlying b-tree record — the de-duplicated PRIMARY KEY
/// columns first (as KEY columns, `key` = 1), then the remaining data columns (in table
/// declaration order, as auxiliary columns, `key` = 0). `cid` is always the column's TABLE
/// ordinal, not its position in the reordered record.
///
/// The KEY columns' order/`desc`/`coll` come from [`primary_key_key_meta`], de-duplicated by
/// [`dedup_pk_key_meta`] so a literal `PRIMARY KEY(a, a)` keys the b-tree on `a` once (spec:
/// "de-duplicated PRIMARY KEY columns"). A DATA column is any column NOT in that PK set, EXCEPT a
/// VIRTUAL generated column: a VIRTUAL value is recomputed on read and occupies no record slot
/// ([`ColumnDef::is_virtual_generated`]), so it is not "used in the record"; a STORED generated
/// column IS materialized in the record, so it is included. Data columns report `desc` = 0,
/// `key` = 0, and their declared `COLLATE` (else the default `BINARY`), via the shared
/// [`index_column_collation`].
///
/// A WITHOUT ROWID table always has a PRIMARY KEY, but a foreign/corrupt schema presenting an
/// empty one yields zero rows (no key columns == no b-tree record shape to describe), matching
/// [`index_info_rows_for_wr_table`], rather than emitting a keyless data-only record.
fn index_xinfo_rows_for_wr_table(tbl: &TableDef) -> Vec<Vec<Value>> {
    debug_assert!(tbl.without_rowid, "WR-table index_xinfo fallback needs WITHOUT ROWID");
    let pk = dedup_pk_key_meta(tbl);
    let mut rows: Vec<Vec<Value>> = Vec::new();
    let mut seqno: usize = 0;
    // KEY columns: the de-duplicated PK columns in PK order, each carrying its own desc/coll.
    for meta in &pk {
        let Some(col) = tbl.columns.get(meta.col_idx) else {
            continue;
        };
        let coll = index_column_collation(meta.collation.as_deref(), Some(tbl), &col.name);
        rows.push(vec![
            Value::Integer(seqno as i64),
            Value::Integer(meta.col_idx as i64),
            Value::Text(col.name.clone()),
            Value::Integer(i64::from(meta.descending)),
            Value::Text(coll),
            Value::Integer(1),
        ]);
        seqno += 1;
    }
    if rows.is_empty() {
        return Vec::new();
    }
    // DATA (auxiliary) columns: every stored non-PK column, in declaration order. A VIRTUAL
    // generated column is skipped (no record slot); a STORED generated column is kept.
    for (idx, col) in tbl.columns.iter().enumerate() {
        if col.is_virtual_generated() || pk.iter().any(|m| m.col_idx == idx) {
            continue;
        }
        let coll = index_column_collation(None, Some(tbl), &col.name);
        rows.push(vec![
            Value::Integer(seqno as i64),
            Value::Integer(idx as i64),
            Value::Text(col.name.clone()),
            Value::Integer(0),
            Value::Text(coll),
            Value::Integer(0),
        ]);
        seqno += 1;
    }
    rows
}

/// The `index_xinfo` rows for an already-resolved index `idx` and its (optional) table `tbl` —
/// the pure core of [`index_xinfo_rows`], testable without a [`Catalog`]. One row per key
/// column (its `seqno`, the `(cid, name)` pair from [`index_column_cid_name`], then
/// `desc`/`coll`/`key`=1), followed by the auxiliary rows from [`index_xinfo_aux_rows`].
///
/// `cid`/`name` follow the shared expression rule ([`index_column_cid_name`]); `desc`/`coll`
/// come from the parallel [`IndexDef::key_columns`] REGARDLESS of whether the key column is an
/// expression, so `CREATE INDEX i ON t(lower(a) COLLATE NOCASE DESC)` still reports
/// `desc`=1 / `coll`="NOCASE" alongside `cid`=-2 / `name`=NULL.
fn index_xinfo_rows_for(idx: &IndexDef, tbl: Option<&TableDef>) -> Vec<Vec<Value>> {
    let mut rows: Vec<Vec<Value>> = idx
        .columns
        .iter()
        .enumerate()
        .map(|(i, colname)| {
            // `key_columns` is parallel to `columns` (a debug-asserted invariant at every
            // construction site), but read it via `get` and fall back to the ASC / inherit
            // defaults rather than panic if a malformed schema ever leaves it shorter than
            // `columns`.
            let key_col = idx.key_columns.get(i);
            let descending = key_col.map(|k| k.descending).unwrap_or(false);
            let coll =
                index_column_collation(key_col.and_then(|k| k.collation.as_deref()), tbl, colname);
            let (cid, name) = index_column_cid_name(idx, tbl, i, colname);
            vec![
                Value::Integer(i as i64),
                Value::Integer(cid),
                name,
                Value::Integer(i64::from(descending)),
                Value::Text(coll),
                Value::Integer(1),
            ]
        })
        .collect();
    // After the key columns come the AUXILIARY columns that locate the table row (a rowid
    // table's rowid, or a WITHOUT ROWID table's not-already-keyed PRIMARY KEY columns); their
    // seqno continues after the key columns.
    rows.extend(index_xinfo_aux_rows(tbl, idx, idx.columns.len()));
    rows
}

/// The AUXILIARY index-record columns `index_xinfo` appends AFTER a secondary index's key
/// columns — the columns needed to locate the table row each index entry points at
/// (pragma.html #pragma_index_xinfo, "Auxiliary columns"). Every returned row has `key` = 0.
///
/// A rowid table appends exactly one: the rowid (`cid` = -1, `name` = NULL, always
/// `desc` = 0 / `coll` = `BINARY`). A WITHOUT ROWID table has no rowid — its rows are keyed
/// by the PRIMARY KEY — so a secondary index instead appends the table's PRIMARY KEY columns
/// that are NOT already present as index key columns, in PRIMARY KEY order (withoutrowid.html;
/// the b-tree record is de-duplicated PRIMARY KEY columns first, then data columns). A PK
/// column already carried as a key column was emitted earlier with `key` = 1 and is skipped
/// here (the de-duplication). Each appended PK column reports its OWN `desc`/`coll` — the
/// PRIMARY KEY's per-column sort/collation, via [`primary_key_key_meta`] — so a
/// `PRIMARY KEY(a DESC, b COLLATE NOCASE)` propagates `desc`/`coll` onto the aux columns.
///
/// `start_seqno` is the rank of the first auxiliary column (the key-column count); the
/// sequence continues from there.
fn index_xinfo_aux_rows(tbl: Option<&TableDef>, idx: &IndexDef, start_seqno: usize) -> Vec<Vec<Value>> {
    let without_rowid = tbl.map(|t| t.without_rowid).unwrap_or(false);
    if !without_rowid {
        // Rowid table: the single rowid auxiliary column (never DESC, always BINARY).
        return vec![vec![
            Value::Integer(start_seqno as i64),
            Value::Integer(-1),
            Value::Null,
            Value::Integer(0),
            Value::Text("BINARY".to_string()),
            Value::Integer(0),
        ]];
    }
    let Some(tbl) = tbl else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    let mut seqno = start_seqno;
    for meta in primary_key_key_meta(tbl) {
        let Some(col) = tbl.columns.get(meta.col_idx) else {
            continue;
        };
        // A PK column that is already an index key column was listed above with key=1;
        // it is not repeated as an auxiliary column (de-duplicated, matching the b-tree).
        if idx.columns.iter().any(|c| c.eq_ignore_ascii_case(&col.name)) {
            continue;
        }
        let coll = index_column_collation(meta.collation.as_deref(), Some(tbl), &col.name);
        rows.push(vec![
            Value::Integer(seqno as i64),
            Value::Integer(meta.col_idx as i64),
            Value::Text(col.name.clone()),
            Value::Integer(i64::from(meta.descending)),
            Value::Text(coll),
            Value::Integer(0),
        ]);
        seqno += 1;
    }
    rows
}

/// One PRIMARY KEY column with the per-column key metadata `index_xinfo` needs to describe it
/// as a WITHOUT ROWID auxiliary index column: its table-column index, its `DESC` flag, and any
/// explicit per-key `COLLATE` override.
struct PkKeyMeta {
    /// The 0-based index of the column within the table.
    col_idx: usize,
    /// The column's `DESC` flag within the PRIMARY KEY (false = ASC).
    descending: bool,
    /// The explicit `COLLATE` written on this column in the PRIMARY KEY declaration, or `None`
    /// when it inherits (resolved later by [`index_column_collation`]: the table column's
    /// declared `COLLATE`, else `BINARY`).
    collation: Option<String>,
}

/// The PRIMARY KEY columns of `tbl`, each with its per-column sort/collation metadata, in
/// PRIMARY KEY order — the source [`index_xinfo_aux_rows`] reads to describe a WITHOUT ROWID
/// table's appended auxiliary PK columns.
///
/// The column identity and order come from [`TableDef::primary_key`] (the single explicit
/// PK-order source). The per-column `descending`/`collation` live on the PRIMARY KEY's
/// [`AutoIndexSpec`](crate::def::AutoIndexSpec), which for a WITHOUT ROWID table is the one
/// spec that reserves its ordinal but owns no separate b-tree (`emit_row == false`; every
/// UNIQUE auto-index has `emit_row == true`, so this never matches a UNIQUE spec). A
/// single-column `INTEGER PRIMARY KEY` reserves NO spec at all, so a PK column with no matching
/// spec entry falls back to ASC / inherit — exactly what a bare integer key column is. Both the
/// create and the load paths populate `auto_indexes`/`primary_key` through
/// `builder::table_def_from_ast`, so this reads correctly for on-disk tables too.
fn primary_key_key_meta(tbl: &TableDef) -> Vec<PkKeyMeta> {
    let pk_spec = tbl.auto_indexes.iter().find(|s| !s.emit_row);
    tbl.primary_key
        .iter()
        .map(|&col_idx| {
            let key = tbl.columns.get(col_idx).and_then(|col| {
                let spec = pk_spec?;
                let pos = spec.columns.iter().position(|c| c.eq_ignore_ascii_case(&col.name))?;
                spec.key_columns.get(pos)
            });
            PkKeyMeta {
                col_idx,
                descending: key.is_some_and(|k| k.descending),
                collation: key.and_then(|k| k.collation.clone()),
            }
        })
        .collect()
}

/// `foreign_key_list(T)` rows — one per column of each FK on `T`, matching SQLite's column
/// set/order: `id, seq, table, from, to, on_update, on_delete, match`.
///
/// SQLite numbers FKs with the LAST-declared constraint getting `id` 0 (its internal list
/// is built by prepending and this pragma walks it head-first), so this iterates
/// [`TableDef::foreign_keys`] (declaration order) in REVERSE. A multi-column FK emits one
/// row per child column (`seq` its 0-based position). `to` is NULL when the FK references
/// the parent's PRIMARY KEY (empty `parent_columns`), and `match` is always `NONE` (SQLite
/// parses but ignores MATCH).
fn foreign_key_list_rows(
    cat: &dyn Catalog,
    schema: Option<&str>,
    name: &str,
) -> Result<Vec<Vec<Value>>> {
    let mut rows = Vec::new();
    if let Some(db) = resolve_table_target(cat, schema, name)? {
        if let Some(tbl) = cat.table_in(db, name)? {
            // id 0 == the LAST-declared FK, so walk declaration order in reverse.
            for (id, fk) in tbl.foreign_keys.iter().rev().enumerate() {
                for (seq, child) in fk.child_columns.iter().enumerate() {
                    // `to` is the matching parent column, or NULL when the FK targets the
                    // parent's PRIMARY KEY (empty parent_columns) — `get(seq)` is None
                    // there, handling both cases uniformly.
                    let to = match fk.parent_columns.get(seq) {
                        Some(col) => Value::Text(col.clone()),
                        None => Value::Null,
                    };
                    rows.push(vec![
                        Value::Integer(id as i64),
                        Value::Integer(seq as i64),
                        Value::Text(fk.parent_table.clone()),
                        Value::Text(child.clone()),
                        to,
                        Value::Text(referential_action_name(fk.on_update).to_string()),
                        Value::Text(referential_action_name(fk.on_delete).to_string()),
                        Value::Text("NONE".to_string()),
                    ]);
                }
            }
        }
    }
    Ok(rows)
}

/// The six shared `table_info` / `table_xinfo` cells for one column: `cid`, `name`, `type`,
/// `notnull`, `dflt_value`, `pk`. `cid` is passed in (the caller decides the numbering — a
/// contiguous result-set rank for `table_info`, the true table index for `table_xinfo`),
/// while `true_idx` is the column's real position, used for `pk` so the key position is
/// correct regardless of how `cid` was assigned.
fn table_info_cells(cid: i64, col: &ColumnDef, tbl: &TableDef, true_idx: usize) -> Vec<Value> {
    vec![
        Value::Integer(cid),
        Value::Text(col.name.clone()),
        // A declared type is stored verbatim; a column with none reports the empty string,
        // not NULL (pragma.html: "data type if given, else ''").
        Value::Text(col.declared_type.clone().unwrap_or_default()),
        Value::Integer(i64::from(col.not_null)),
        match &col.default {
            Some(d) => Value::Text(d.clone()),
            None => Value::Null,
        },
        Value::Integer(table_info_pk(tbl, true_idx)),
    ]
}

/// The `hidden` flag `table_xinfo` reports (pragma.html): 0 ordinary, 2 VIRTUAL generated,
/// 3 STORED generated. Flag 1 (a virtual-table hidden column) is never produced — this
/// engine stores only regular tables.
fn column_hidden_flag(col: &ColumnDef) -> i64 {
    match &col.generated {
        Some(g) if g.stored => 3,
        Some(_) => 2,
        None => 0,
    }
}

/// The `pk` value `table_info` reports for column `col_idx`: 0 when the column is not part
/// of the primary key, else its 1-based position WITHIN the primary key, in PRIMARY KEY
/// declaration order (pragma.html, table_info: "the 1-based index of the column within the
/// primary key").
///
/// Read directly from [`TableDef::primary_key`], the single explicit source of PK-column
/// order the builder records for EVERY PK form uniformly — an `INTEGER PRIMARY KEY` rowid
/// alias, a column-level `PRIMARY KEY`, and a table-level `PRIMARY KEY(c1, c2, …)`
/// (composite). A single-column key reports 1; a composite key reports 1, 2, … across its
/// members in declaration order.
fn table_info_pk(tbl: &TableDef, col_idx: usize) -> i64 {
    tbl.primary_key.iter().position(|&i| i == col_idx).map_or(0, |p| (p + 1) as i64)
}

/// The `origin` string `index_list` reports: `'c'` for a `CREATE INDEX`, `'pk'`/`'u'` for a
/// PRIMARY KEY / UNIQUE constraint index (fileformat2 §2.6.2).
///
/// The reserved `sqlite_autoindex_` name is given only to constraint-backed indexes, so its
/// presence separates `'c'` from a constraint index. Within the constraint case, the PRIMARY
/// KEY auto-index is the one whose key columns ARE the table's primary key — compared against
/// [`TableDef::primary_key`] by column name, in PK order (see [`index_is_primary_key`]) —
/// while every OTHER constraint auto-index is a UNIQUE one. This covers a table-level /
/// composite PRIMARY KEY uniformly, not just a column-level one. (A WITHOUT ROWID table's PK
/// is the table b-tree itself, not a separate `sqlite_autoindex_`, so `index_list` there only
/// ever sees its UNIQUE auto-indexes — correctly reported `'u'`.)
fn index_origin(idx: &IndexDef, tbl: Option<&TableDef>) -> &'static str {
    if !idx.name.to_ascii_lowercase().starts_with("sqlite_autoindex_") {
        return "c";
    }
    match tbl {
        Some(tbl) if index_is_primary_key(idx, tbl) => "pk",
        _ => "u",
    }
}

/// Whether `idx`'s key columns are exactly `tbl`'s PRIMARY KEY columns, in order — the test
/// that separates the PRIMARY KEY `sqlite_autoindex_` from a UNIQUE one in [`index_origin`].
///
/// Both [`TableDef::primary_key`] (mapped to column names) and an auto-index's `columns` are
/// in PRIMARY KEY declaration order, so an ordered, case-insensitive name comparison is
/// exact. An empty primary key (no declared PK) matches nothing, so a `sqlite_autoindex_` on
/// such a table is always a UNIQUE index.
///
/// LIMITATION (out of scope here; a known edge for future PK hardening): identity is by column NAME + order only. A redundant `UNIQUE` whose key
/// columns equal the PK's in the SAME order (`PRIMARY KEY(a, b), UNIQUE(a, b)`) would also
/// match and be mislabeled `'pk'`. A `UNIQUE` in a DIFFERENT order (`UNIQUE(b, a)`) is
/// correctly `'u'` (pinned by `index_origin_pk_match_is_order_sensitive_not_set`). The
/// auto-index spec model carries no durable PK marker, so disambiguating the same-order case
/// needs a spec-level `is_pk` flag rather than a column comparison.
fn index_is_primary_key(idx: &IndexDef, tbl: &TableDef) -> bool {
    if tbl.primary_key.is_empty() || idx.columns.len() != tbl.primary_key.len() {
        return false;
    }
    idx.columns.iter().zip(&tbl.primary_key).all(|(name, &pk_idx)| {
        tbl.columns.get(pk_idx).is_some_and(|c| c.name.eq_ignore_ascii_case(name))
    })
}

/// The action name `foreign_key_list` prints for a [`ReferentialAction`], exactly as SQLite
/// spells it: the omitted-action default `NO ACTION`, plus `CASCADE`, `SET NULL`,
/// `SET DEFAULT`, `RESTRICT`. Exhaustive (no wildcard) so a new action variant is a compile
/// error here, not a silently wrong name.
fn referential_action_name(action: ReferentialAction) -> &'static str {
    match action {
        ReferentialAction::NoAction => "NO ACTION",
        ReferentialAction::Restrict => "RESTRICT",
        ReferentialAction::SetNull => "SET NULL",
        ReferentialAction::SetDefault => "SET DEFAULT",
        ReferentialAction::Cascade => "CASCADE",
    }
}

/// The `(cid, name)` pair `index_info` / `index_xinfo` report for key column `i` of `idx`
/// (pragma.html #pragma_index_info / #pragma_index_xinfo output columns 2 and 3). The single
/// place this rule is decided, so the two pragmas cannot drift on it.
///
/// A GENUINE EXPRESSION key column — `CREATE INDEX i ON t(a+1)` / `t(lower(a))`
/// (lang_createindex.html §1.2), recorded as `key_exprs[i] == Some(expr)` with an empty-name
/// sentinel in `columns[i]` (see [`crate::builder`]) — reports `cid = -2` and `name = NULL`
/// ("A value of ... -2 means that an expression is being used"; name "NULL if the column is
/// the rowid or an expression"). An ordinary NAMED key column reports its 0-based table
/// ordinal via [`column_ordinal`] (or -1 when unresolvable) and its column name.
///
/// `key_exprs` is read via `.get(i)` (never a direct index) so a malformed or foreign schema
/// whose `key_exprs` is shorter than `columns` falls back to the ordinary named-column path
/// rather than panicking — the same defensive read the callers use for the parallel
/// `key_columns`. The rowid auxiliary column is NOT handled here (it is not one of `idx`'s key
/// columns); [`index_xinfo_aux_rows`] emits it with `cid` = -1 / `name` = NULL directly.
fn index_column_cid_name(
    idx: &IndexDef,
    tbl: Option<&TableDef>,
    i: usize,
    colname: &str,
) -> (i64, Value) {
    if matches!(idx.key_exprs.get(i), Some(Some(_))) {
        (-2, Value::Null)
    } else {
        (column_ordinal(tbl, colname), Value::Text(colname.to_string()))
    }
}

/// The 0-based ordinal of `colname` within `tbl`'s columns, or -1 when the table is absent
/// or the column is not a named table column (the `index_info`/`index_xinfo` sentinel for a
/// rowid/expression column).
fn column_ordinal(tbl: Option<&TableDef>, colname: &str) -> i64 {
    match tbl {
        Some(t) => t
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(colname))
            .map(|p| p as i64)
            .unwrap_or(-1),
        None => -1,
    }
}

/// The collation name `index_xinfo` reports for one key column, resolved by SQLite's
/// collation inheritance chain (datatype3.html §7.1): an explicit per-key `COLLATE` name,
/// else the target table column's declared `COLLATE`, else the default `BINARY`.
///
/// Reported VERBATIM in the exact case written in the SQL (real sqlite echoes the `COLLATE`
/// token unchanged); the upper-case `BINARY` is reported ONLY for the implicit default. So
/// DO NOT normalize case — doing so diverges byte-for-byte for any non-upper-case collation.
fn index_column_collation(explicit: Option<&str>, tbl: Option<&TableDef>, colname: &str) -> String {
    explicit.or_else(|| column_declared_collation(tbl, colname)).unwrap_or("BINARY").to_string()
}

/// The declared `COLLATE` of the column named `colname` on `tbl`, or `None` when the
/// table/column is absent or the column declares no collation.
fn column_declared_collation<'a>(tbl: Option<&'a TableDef>, colname: &str) -> Option<&'a str> {
    tbl?.columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(colname))?
        .collation
        .as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::table_def_from_ast;
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop, Statement};

    /// Build a [`TableDef`] from a single `CREATE TABLE` via the real builder, so
    /// `primary_key` is populated by production code rather than a hand-set literal.
    fn tdef(sql: &str) -> TableDef {
        let ast = parse(sql).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        match ast.statements.as_slice() {
            [Statement::CreateTable(ct)] => table_def_from_ast(ct, 2).unwrap(),
            other => panic!("expected one CREATE TABLE, got {other:?}"),
        }
    }

    /// The [`IndexDef`] for the table's Nth auto-index spec (built through the real
    /// `from_auto_spec` derivation the create/load paths use).
    fn auto_index(tbl: &TableDef, n: usize) -> IndexDef {
        let spec = tbl.auto_indexes.get(n).expect("expected auto-index at n");
        IndexDef::from_auto_spec(spec.name.clone(), tbl.name.clone(), spec, 3)
    }

    #[test]
    fn table_info_pk_reports_composite_position_in_declaration_order() {
        // PRIMARY KEY(b, a): b is the 1st key column, a the 2nd, c is not in the key.
        let t = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(b, a))");
        assert_eq!(table_info_pk(&t, 0), 2, "a is the 2nd PK column");
        assert_eq!(table_info_pk(&t, 1), 1, "b is the 1st PK column");
        assert_eq!(table_info_pk(&t, 2), 0, "c is not part of the PK");
    }

    #[test]
    fn index_list_origin_is_pk_for_composite_auto_index() {
        let t = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(b, a))");
        let idx = auto_index(&t, 0);
        assert!(idx.name.starts_with("sqlite_autoindex_"), "auto-index name: {}", idx.name);
        assert_eq!(index_origin(&idx, Some(&t)), "pk");
    }

    #[test]
    fn single_column_column_level_pk_has_no_regression() {
        // A non-integer column-level PK reports pk=1 and its auto-index origin 'pk'. (An
        // INTEGER PK would be the rowid alias and own no separate auto-index.)
        let t = tdef("CREATE TABLE t(id TEXT PRIMARY KEY, x)");
        assert_eq!(table_info_pk(&t, 0), 1);
        assert_eq!(table_info_pk(&t, 1), 0);
        assert_eq!(index_origin(&auto_index(&t, 0), Some(&t)), "pk");
    }

    #[test]
    fn unique_auto_index_is_u_and_create_index_is_c() {
        // A UNIQUE constraint's auto-index over a non-PK column is 'u', not 'pk', even
        // alongside a composite PRIMARY KEY; a user `CREATE INDEX` (non-`sqlite_autoindex_`
        // name) is 'c'.
        let t = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(b, a), UNIQUE(c))");
        let uq_spec =
            t.auto_indexes.iter().find(|s| s.columns == ["c"]).expect("unique auto-index over c");
        let uq = IndexDef::from_auto_spec(uq_spec.name.clone(), t.name.clone(), uq_spec, 4);
        assert_eq!(index_origin(&uq, Some(&t)), "u");
        let mut user_index = uq.clone();
        user_index.name = "ix_c".to_string();
        assert_eq!(index_origin(&user_index, Some(&t)), "c");
    }

    #[test]
    fn index_origin_pk_match_is_order_sensitive_not_set() {
        // A UNIQUE over the SAME columns as the composite PK but in a DIFFERENT order is a
        // distinct index real sqlite keeps; its origin is 'u', not 'pk'. This pins the ORDERED
        // name comparison in `index_is_primary_key`: a set-only compare would wrongly tag
        // `UNIQUE(b, a)` as 'pk' because {a, b} == {b, a}.
        let t = tdef("CREATE TABLE t(a, b, PRIMARY KEY(a, b), UNIQUE(b, a))");
        // Two distinct auto-indexes, in declaration order: PK(a, b) first, UNIQUE(b, a) second.
        assert_eq!(t.auto_indexes[0].columns, ["a", "b"], "PK auto-index keeps PK order");
        assert_eq!(t.auto_indexes[1].columns, ["b", "a"], "UNIQUE auto-index keeps its own order");
        assert_eq!(index_origin(&auto_index(&t, 0), Some(&t)), "pk", "PK(a, b) auto-index");
        assert_eq!(
            index_origin(&auto_index(&t, 1), Some(&t)),
            "u",
            "UNIQUE(b, a) is a different index (columns in a different order) -> not the PK"
        );
    }

    #[test]
    fn primary_key_key_meta_reads_composite_pk_order_sort_and_collation() {
        // A WITHOUT ROWID composite PK carries per-column DESC/COLLATE on its (emit_row=false)
        // auto-index spec. `primary_key_key_meta` surfaces it in PRIMARY KEY order (a, then b),
        // which is what the aux-column builder appends. `a DESC` -> descending; `b COLLATE
        // NOCASE` -> the explicit override (a's is None: it inherits).
        let t =
            tdef("CREATE TABLE w(a, b, c, PRIMARY KEY(a DESC, b COLLATE NOCASE)) WITHOUT ROWID");
        let meta = primary_key_key_meta(&t);
        assert_eq!(meta.len(), 2, "two PK columns, non-members excluded");
        assert_eq!(meta[0].col_idx, 0);
        assert!(meta[0].descending, "a DESC -> descending");
        assert_eq!(meta[0].collation, None, "a inherits (no explicit COLLATE)");
        assert_eq!(meta[1].col_idx, 1);
        assert!(!meta[1].descending, "b is ASC");
        assert_eq!(meta[1].collation.as_deref(), Some("NOCASE"), "b's explicit COLLATE override");
    }

    #[test]
    fn primary_key_key_meta_declaration_order_not_column_order() {
        // The metadata follows PRIMARY KEY declaration order, not table-column order:
        // `PRIMARY KEY(b, a)` yields b (col 1) first, then a (col 0).
        let t = tdef("CREATE TABLE w(a, b, PRIMARY KEY(b, a)) WITHOUT ROWID");
        let meta = primary_key_key_meta(&t);
        assert_eq!(meta.iter().map(|m| m.col_idx).collect::<Vec<_>>(), vec![1, 0]);
    }

    #[test]
    fn primary_key_key_meta_plain_integer_pk_has_no_spec_falls_back_asc_inherit() {
        // A single-column `INTEGER PRIMARY KEY` reserves NO auto-index spec at all (it is the
        // key with no separate b-tree AND no reserved ordinal), so the metadata must fall back
        // to ASC / inherit rather than panic or drop the column.
        let t = tdef("CREATE TABLE w(k INTEGER PRIMARY KEY, v) WITHOUT ROWID");
        assert!(t.auto_indexes.iter().all(|s| s.emit_row), "integer PK reserves no spec");
        let meta = primary_key_key_meta(&t);
        assert_eq!(meta.len(), 1);
        assert_eq!(meta[0].col_idx, 0);
        assert!(!meta[0].descending);
        assert_eq!(meta[0].collation, None);
    }

    #[test]
    fn index_xinfo_aux_rows_rowid_table_is_the_single_rowid() {
        // A rowid table's auxiliary column is the rowid: seqno continues, cid=-1, name=NULL,
        // desc=0, coll='BINARY', key=0.
        let t = tdef("CREATE TABLE t(a UNIQUE, b)");
        let uq = auto_index(&t, 0);
        let aux = index_xinfo_aux_rows(Some(&t), &uq, 1);
        assert_eq!(aux.len(), 1, "one rowid auxiliary column");
        assert_row(&aux[0], 1, -1, None, 0, "BINARY", 0);
    }

    #[test]
    fn index_xinfo_aux_rows_without_rowid_appends_not_already_keyed_pk_columns() {
        // A UNIQUE(c) secondary index on a WITHOUT ROWID table keyed by PRIMARY KEY(a, b):
        // after the key column c, index_xinfo appends the PK columns a, b (key=0) in PK order.
        let t = tdef("CREATE TABLE w(a, b, c, PRIMARY KEY(a, b), UNIQUE(c)) WITHOUT ROWID");
        let uq_spec =
            t.auto_indexes.iter().find(|s| s.columns == ["c"]).expect("UNIQUE(c) auto-index");
        let uq = IndexDef::from_auto_spec(uq_spec.name.clone(), t.name.clone(), uq_spec, 5);
        let aux = index_xinfo_aux_rows(Some(&t), &uq, 1);
        assert_eq!(aux.len(), 2, "two aux PK columns a, b");
        assert_row(&aux[0], 1, 0, Some("a"), 0, "BINARY", 0);
        assert_row(&aux[1], 2, 1, Some("b"), 0, "BINARY", 0);
    }

    #[test]
    fn index_xinfo_aux_rows_without_rowid_dedups_a_pk_column_already_in_the_key() {
        // When a secondary index key already contains a PK column, that column is NOT repeated
        // as an auxiliary column (de-duplication). PRIMARY KEY(a, b); a UNIQUE-style index over
        // (c, a) keeps a as a key column, so only b is appended as an auxiliary column.
        let t = tdef("CREATE TABLE w(a, b, c, PRIMARY KEY(a, b)) WITHOUT ROWID");
        // Build a two-column index (c, a) via the real CREATE INDEX builder.
        let idx = idef("CREATE INDEX wi ON w(c, a)", &t);
        // Key columns c, a occupy seqno 0,1 -> aux starts at 2. Only b remains.
        let aux = index_xinfo_aux_rows(Some(&t), &idx, 2);
        assert_eq!(aux.len(), 1, "a is de-duplicated; only b is appended");
        assert_row(&aux[0], 2, 1, Some("b"), 0, "BINARY", 0);
    }

    #[test]
    fn index_xinfo_aux_rows_without_rowid_propagates_pk_desc_and_collation() {
        // The appended aux PK columns carry the PRIMARY KEY's per-column desc/coll:
        // PRIMARY KEY(a DESC, b COLLATE NOCASE) -> aux a desc=1/BINARY, b desc=0/NOCASE.
        let t =
            tdef("CREATE TABLE w(a, b, c, PRIMARY KEY(a DESC, b COLLATE NOCASE)) WITHOUT ROWID");
        let idx = idef("CREATE INDEX wi ON w(c)", &t);
        let aux = index_xinfo_aux_rows(Some(&t), &idx, 1);
        assert_eq!(aux.len(), 2);
        assert_row(&aux[0], 1, 0, Some("a"), 1, "BINARY", 0);
        assert_row(&aux[1], 2, 1, Some("b"), 0, "NOCASE", 0);
    }

    /// Assert one `index_xinfo` row cell-by-cell (`Value` has no `PartialEq`). `name` is
    /// `Some` for a named column or `None` for the rowid's NULL name.
    fn assert_row(
        row: &[Value],
        seqno: i64,
        cid: i64,
        name: Option<&str>,
        desc: i64,
        coll: &str,
        key: i64,
    ) {
        assert_eq!(row.len(), 6, "index_xinfo row is 6 cells wide");
        assert!(matches!(&row[0], Value::Integer(v) if *v == seqno), "seqno: {:?}", row[0]);
        assert!(matches!(&row[1], Value::Integer(v) if *v == cid), "cid: {:?}", row[1]);
        match name {
            Some(n) => assert!(matches!(&row[2], Value::Text(t) if t == n), "name: {:?}", row[2]),
            None => assert!(matches!(&row[2], Value::Null), "name should be NULL: {:?}", row[2]),
        }
        assert!(matches!(&row[3], Value::Integer(v) if *v == desc), "desc: {:?}", row[3]);
        assert!(matches!(&row[4], Value::Text(t) if t == coll), "coll: {:?}", row[4]);
        assert!(matches!(&row[5], Value::Integer(v) if *v == key), "key: {:?}", row[5]);
    }

    /// Build the [`IndexDef`] for a single `CREATE INDEX` via the real builder, resolved
    /// against table `tbl` (so `cid`s and dedup match production).
    fn idef(sql: &str, tbl: &TableDef) -> IndexDef {
        let ast = parse(sql).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        match ast.statements.as_slice() {
            [Statement::CreateIndex(ci)] => {
                crate::builder::index_def_from_ast(ci, &tbl.name, &tbl.columns, 9).unwrap()
            }
            other => panic!("expected one CREATE INDEX, got {other:?}"),
        }
    }

    // --- expression key columns report cid=-2 / name=NULL (pragma.html) ----------

    #[test]
    fn index_info_expression_key_reports_cid_minus_two_and_null_name() {
        // A GENUINE expression index `CREATE INDEX i ON t(a + 1)` over rowid table t(a, b):
        // index_info's one row is the expression key column, which reports cid=-2 (an
        // expression is being used) and name=NULL — NOT the buggy cid=-1 / name="" the
        // empty-name sentinel would fall through to (pragma.html #pragma_index_info, cols 2/3).
        let t = tdef("CREATE TABLE t(a, b)");
        let idx = idef("CREATE INDEX i ON t(a + 1)", &t);
        let rows = index_info_rows_for(&idx, Some(&t));
        assert_eq!(rows.len(), 1, "one key column");
        assert_info_row(&rows[0], 0, -2, None);
    }

    #[test]
    fn index_xinfo_expression_key_over_rowid_table_then_rowid_aux() {
        // index_xinfo on the same expression index over a ROWID table: the expression key row
        // reports cid=-2/name=NULL but keeps desc=0/coll=BINARY/key=1, followed by the
        // UNCHANGED rowid auxiliary row (cid=-1/name=NULL/key=0).
        let t = tdef("CREATE TABLE t(a, b)");
        let idx = idef("CREATE INDEX i ON t(a + 1)", &t);
        let rows = index_xinfo_rows_for(&idx, Some(&t));
        assert_eq!(rows.len(), 2, "one key column + the rowid aux column");
        assert_row(&rows[0], 0, -2, None, 0, "BINARY", 1);
        assert_row(&rows[1], 1, -1, None, 0, "BINARY", 0);
    }

    #[test]
    fn index_info_mixed_named_and_expression_columns_is_per_column() {
        // A MIXED index `CREATE INDEX i ON t(a, b + 1)`: the ordinary column a keeps its
        // ordinal/name, while the expression b + 1 gets cid=-2/name=NULL. Proves the rule is
        // decided PER COLUMN via key_exprs[i], not all-or-nothing for the whole index.
        let t = tdef("CREATE TABLE t(a, b)");
        let idx = idef("CREATE INDEX i ON t(a, b + 1)", &t);
        let rows = index_info_rows_for(&idx, Some(&t));
        assert_eq!(rows.len(), 2);
        assert_info_row(&rows[0], 0, 0, Some("a"));
        assert_info_row(&rows[1], 1, -2, None);
    }

    #[test]
    fn index_xinfo_expression_key_still_carries_collate_and_desc() {
        // An expression key with a trailing COLLATE/DESC still reports cid=-2/name=NULL, but
        // its desc/coll come through from the parallel key_columns entry: `lower(a) COLLATE
        // NOCASE DESC` -> desc=1, coll="NOCASE". ONLY cid/name change for an expression column.
        let t = tdef("CREATE TABLE t(a)");
        let idx = idef("CREATE INDEX i ON t(lower(a) COLLATE NOCASE DESC)", &t);
        let rows = index_xinfo_rows_for(&idx, Some(&t));
        assert_eq!(rows.len(), 2, "one key column + the rowid aux column");
        assert_row(&rows[0], 0, -2, None, 1, "NOCASE", 1);
        assert_row(&rows[1], 1, -1, None, 0, "BINARY", 0);
    }

    #[test]
    fn index_info_ordinary_named_index_is_unchanged() {
        // Regression: an ordinary named-column index `CREATE INDEX i ON t(b, a)` is UNCHANGED
        // — cid is the table ordinal, name the column name (the expression path must not leak
        // into the ordinary path).
        let t = tdef("CREATE TABLE t(a, b)");
        let idx = idef("CREATE INDEX i ON t(b, a)", &t);
        let rows = index_info_rows_for(&idx, Some(&t));
        assert_eq!(rows.len(), 2);
        assert_info_row(&rows[0], 0, 1, Some("b"));
        assert_info_row(&rows[1], 1, 0, Some("a"));
    }

    #[test]
    fn key_exprs_shorter_than_columns_falls_back_to_ordinary_and_never_panics() {
        // GUARDRAIL: a malformed or foreign `IndexDef` whose `key_exprs` is SHORTER than
        // `columns` must fall back to the ordinary named-column path via the defensive
        // `.get(i)` in `index_column_cid_name` and NEVER panic — the same contract the
        // desc/coll read upholds on `key_columns`. The real builder always keeps the parallel
        // vectors in lockstep, so this branch is otherwise unreachable; hand-clearing
        // `key_exprs` (leaving `columns`/`key_columns` intact) is the only way to exercise it.
        // With no expression signal, every column reports its ordinary ordinal/name.
        let t = tdef("CREATE TABLE t(a, b)");
        let mut idx = idef("CREATE INDEX i ON t(a, b)", &t);
        idx.key_exprs.clear();
        let info = index_info_rows_for(&idx, Some(&t));
        assert_eq!(info.len(), 2);
        assert_info_row(&info[0], 0, 0, Some("a"));
        assert_info_row(&info[1], 1, 1, Some("b"));
        let xinfo = index_xinfo_rows_for(&idx, Some(&t));
        assert_eq!(xinfo.len(), 3, "two key columns + the rowid aux column");
        assert_row(&xinfo[0], 0, 0, Some("a"), 0, "BINARY", 1);
        assert_row(&xinfo[1], 1, 1, Some("b"), 0, "BINARY", 1);
        assert_row(&xinfo[2], 2, -1, None, 0, "BINARY", 0);
    }

    #[test]
    fn index_xinfo_expression_key_on_without_rowid_table_then_pk_aux() {
        // The expression fix composes with the WITHOUT ROWID aux-column enumeration: the
        // expression key row reports cid=-2/name=NULL/key=1, then the table's PRIMARY KEY
        // column `a` (not one of the index's key columns — the key is the expression `b + 1`)
        // is appended as an auxiliary column (cid=0/name="a"/key=0). No rowid aux row, because
        // a WITHOUT ROWID table has no rowid.
        let t = tdef("CREATE TABLE w(a, b, PRIMARY KEY(a)) WITHOUT ROWID");
        let idx = idef("CREATE INDEX i ON w(b + 1)", &t);
        let rows = index_xinfo_rows_for(&idx, Some(&t));
        assert_eq!(rows.len(), 2, "one expression key column + the PK aux column a");
        assert_row(&rows[0], 0, -2, None, 0, "BINARY", 1);
        assert_row(&rows[1], 1, 0, Some("a"), 0, "BINARY", 0);
    }

    // --- WITHOUT ROWID TABLE fallback for index_info / index_xinfo (pragma.html, 3.30.0) ----

    #[test]
    fn index_info_wr_table_is_dedup_pk_columns_in_pk_order() {
        // pragma.html #pragma_index_info: a WITHOUT ROWID table named where an index is
        // expected yields its PRIMARY KEY columns (as they key the b-tree). One row per PK
        // column, cid = TABLE ordinal, in PK order — c (not in the PK) is absent.
        let t = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(a, b)) WITHOUT ROWID");
        let rows = index_info_rows_for_wr_table(&t);
        assert_eq!(rows.len(), 2, "two PK columns; the data column c is not a key column");
        assert_info_row(&rows[0], 0, 0, Some("a"));
        assert_info_row(&rows[1], 1, 1, Some("b"));
    }

    #[test]
    fn index_info_wr_table_order_is_pk_declaration_order_not_table_order() {
        // PRIMARY KEY(b, a): the key order is b then a, NOT the table column order a then b —
        // seqno follows the PRIMARY KEY declaration, and cid is each column's table ordinal.
        let t = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(b, a)) WITHOUT ROWID");
        let rows = index_info_rows_for_wr_table(&t);
        assert_eq!(rows.len(), 2);
        assert_info_row(&rows[0], 0, 1, Some("b"));
        assert_info_row(&rows[1], 1, 0, Some("a"));
    }

    #[test]
    fn index_xinfo_wr_table_is_pk_key_columns_then_data_columns() {
        // pragma.html #pragma_index_xinfo: the b-tree record — de-duplicated PK columns first
        // (key=1), then the remaining data columns in declaration order (key=0). PK(a, b) then
        // data c. Every column here inherits BINARY / ASC.
        let t = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(a, b)) WITHOUT ROWID");
        let rows = index_xinfo_rows_for_wr_table(&t);
        assert_eq!(rows.len(), 3, "two PK key columns + one data column");
        assert_row(&rows[0], 0, 0, Some("a"), 0, "BINARY", 1);
        assert_row(&rows[1], 1, 1, Some("b"), 0, "BINARY", 1);
        assert_row(&rows[2], 2, 2, Some("c"), 0, "BINARY", 0);
    }

    #[test]
    fn index_xinfo_wr_table_key_columns_carry_pk_desc_and_collation() {
        // The KEY (PK) columns carry the PRIMARY KEY's per-column desc/coll (verbatim case):
        // `a DESC` -> desc=1 (inherits BINARY), `b COLLATE NOCASE` -> coll="NOCASE". Both
        // columns are in the PK, so there are no trailing data columns.
        let t = tdef("CREATE TABLE t(a, b, PRIMARY KEY(a DESC, b COLLATE NOCASE)) WITHOUT ROWID");
        let rows = index_xinfo_rows_for_wr_table(&t);
        assert_eq!(rows.len(), 2, "both columns are PK key columns; no data columns");
        assert_row(&rows[0], 0, 0, Some("a"), 1, "BINARY", 1);
        assert_row(&rows[1], 1, 1, Some("b"), 0, "NOCASE", 1);
    }

    #[test]
    fn index_xinfo_wr_table_omits_virtual_generated_keeps_stored_generated_data_columns() {
        // A DATA column is any stored non-PK column: a VIRTUAL generated column occupies no
        // b-tree record slot (omit it), a STORED generated column is materialized (keep it).
        // Columns: a(0, PK) b(1) v(2, VIRTUAL) s(3, STORED). Record: a | b, s (v dropped).
        let t = tdef(
            "CREATE TABLE t(a, b, v AS (a + b) VIRTUAL, s AS (a + b) STORED, PRIMARY KEY(a)) \
             WITHOUT ROWID",
        );
        let rows = index_xinfo_rows_for_wr_table(&t);
        assert_eq!(rows.len(), 3, "PK a (key) + data b + data s; the VIRTUAL column v is omitted");
        assert_row(&rows[0], 0, 0, Some("a"), 0, "BINARY", 1);
        assert_row(&rows[1], 1, 1, Some("b"), 0, "BINARY", 0);
        assert_row(&rows[2], 2, 3, Some("s"), 0, "BINARY", 0);
        // index_info reports only the key column, unaffected by the data/generated columns.
        let info = index_info_rows_for_wr_table(&t);
        assert_eq!(info.len(), 1);
        assert_info_row(&info[0], 0, 0, Some("a"));
    }

    #[test]
    fn wr_table_helpers_are_empty_for_a_pk_less_table_and_never_panic() {
        // Defensive: a WITHOUT ROWID table always has a PK (create-time validation), but a
        // foreign/corrupt schema could present none. Both helpers must yield zero rows (no key
        // columns == no b-tree record shape) rather than panicking or emitting a keyless record.
        let mut t = tdef("CREATE TABLE t(a, b, PRIMARY KEY(a)) WITHOUT ROWID");
        t.primary_key.clear();
        assert!(index_info_rows_for_wr_table(&t).is_empty());
        assert!(index_xinfo_rows_for_wr_table(&t).is_empty());
    }

    #[test]
    fn wr_table_dedups_a_repeated_pk_column() {
        // `PRIMARY KEY(a, a)` WITHOUT ROWID is accepted by the builder (duplicate PK COLUMNS are
        // not rejected -> `primary_key == [0, 0]`), but the b-tree keys on the DE-DUPLICATED PK
        // (the executor's `wr_layout` stores `a` once). pragma.html:1113-1114 ("duplicate columns
        // removed") / :1165-1166 ("de-duplicated PRIMARY KEY columns first ...") pin the pragma to
        // match the record: index_info -> 1 row (a), index_xinfo -> a (key) + b (data) = 2 rows,
        // NOT a two-slot key `a, a`.
        let t = tdef("CREATE TABLE t(a, b, PRIMARY KEY(a, a)) WITHOUT ROWID");
        assert_eq!(t.primary_key, vec![0, 0], "builder keeps the repeated PK column");
        let info = index_info_rows_for_wr_table(&t);
        assert_eq!(info.len(), 1, "de-duplicated PK: `a` appears once");
        assert_info_row(&info[0], 0, 0, Some("a"));
        let xinfo = index_xinfo_rows_for_wr_table(&t);
        assert_eq!(xinfo.len(), 2, "a (key) + b (data), not a, a, b");
        assert_row(&xinfo[0], 0, 0, Some("a"), 0, "BINARY", 1);
        assert_row(&xinfo[1], 1, 1, Some("b"), 0, "BINARY", 0);
    }

    #[test]
    fn index_xinfo_wr_table_data_column_carries_its_declared_collation() {
        // A DATA (non-PK) column reports its DECLARED collation, not a hardcoded BINARY: a
        // non-key `b COLLATE NOCASE` -> coll="NOCASE" (spec: data col coll = declared-else-BINARY).
        // Distinguishes reading `col.collation` from a constant "BINARY" that would pass every
        // all-uncollated data column elsewhere in the suite.
        let t = tdef("CREATE TABLE t(a, b COLLATE NOCASE, PRIMARY KEY(a)) WITHOUT ROWID");
        let rows = index_xinfo_rows_for_wr_table(&t);
        assert_eq!(rows.len(), 2, "PK a (key) + data b");
        assert_row(&rows[0], 0, 0, Some("a"), 0, "BINARY", 1);
        assert_row(&rows[1], 1, 1, Some("b"), 0, "NOCASE", 0);
    }

    #[test]
    fn index_info_and_xinfo_are_empty_for_a_rowid_table_name() {
        // The 3.30.0 fallback is WITHOUT-ROWID-SPECIFIC: a ROWID table named where an index is
        // expected yields NO rows (its key is the rowid, not a PK record). Exercised through the
        // real name-resolution wiring, which guards on `without_rowid`.
        let t = tdef("CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        assert!(!t.without_rowid, "sanity: a plain INTEGER PRIMARY KEY table is a rowid table");
        let cat = TestCatalog { tables: vec![t], indexes: vec![] };
        assert!(index_info_rows(&cat, None, "t").unwrap().is_empty());
        assert!(index_xinfo_rows(&cat, None, "t").unwrap().is_empty());
    }

    #[test]
    fn index_info_and_xinfo_are_empty_for_an_unknown_name() {
        // A name that is neither an index nor a table yields zero rows (both surfaces).
        let cat = TestCatalog { tables: vec![], indexes: vec![] };
        assert!(index_info_rows(&cat, None, "nope").unwrap().is_empty());
        assert!(index_xinfo_rows(&cat, None, "nope").unwrap().is_empty());
    }

    #[test]
    fn index_info_resolves_a_wr_table_name_case_insensitively() {
        // SQLite identifiers are case-insensitive: `index_info('T')` resolves the table `t` and
        // returns its PK rows, going through the real resolve_table_target -> table_in path.
        let t = tdef("CREATE TABLE t(a, b, PRIMARY KEY(a, b)) WITHOUT ROWID");
        let cat = TestCatalog { tables: vec![t], indexes: vec![] };
        let rows = index_info_rows(&cat, None, "T").unwrap();
        assert_eq!(rows.len(), 2);
        assert_info_row(&rows[0], 0, 0, Some("a"));
        assert_info_row(&rows[1], 1, 1, Some("b"));
    }

    #[test]
    fn a_real_index_name_still_uses_the_index_path_not_the_table_fallback() {
        // Regression: an actual index name must resolve via the INDEX path (tried first),
        // unchanged by the new table fallback. `ix ON t(b, a)` over a rowid table reports its
        // key columns b, a, then the rowid auxiliary column — exactly the pre-existing behavior.
        let t = tdef("CREATE TABLE t(a, b)");
        let ix = idef("CREATE INDEX ix ON t(b, a)", &t);
        let cat = TestCatalog { tables: vec![t], indexes: vec![ix] };
        let info = index_info_rows(&cat, None, "ix").unwrap();
        assert_eq!(info.len(), 2);
        assert_info_row(&info[0], 0, 1, Some("b"));
        assert_info_row(&info[1], 1, 0, Some("a"));
        let xinfo = index_xinfo_rows(&cat, None, "ix").unwrap();
        assert_eq!(xinfo.len(), 3, "two key columns + the rowid aux column");
        assert_row(&xinfo[0], 0, 1, Some("b"), 0, "BINARY", 1);
        assert_row(&xinfo[1], 1, 0, Some("a"), 0, "BINARY", 1);
        assert_row(&xinfo[2], 2, -1, None, 0, "BINARY", 0);
    }

    /// A minimal in-memory [`Catalog`] for exercising the NAME-RESOLUTION wiring of
    /// `index_info_rows` / `index_xinfo_rows` (index lookup first, then the WITHOUT ROWID table
    /// fallback) without a pager. Only the read lookups those two functions reach carry real
    /// data; every namespace-qualified seam keeps its single-store default, and the write path
    /// is `unreachable!` because introspection never invokes it.
    struct TestCatalog {
        tables: Vec<TableDef>,
        indexes: Vec<IndexDef>,
    }

    impl Catalog for TestCatalog {
        fn table(&self, name: &str) -> Result<Option<&TableDef>> {
            Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
        }
        fn index(&self, name: &str) -> Result<Option<&IndexDef>> {
            Ok(self.indexes.iter().find(|i| i.name.eq_ignore_ascii_case(name)))
        }
        fn indexes_on<'a>(&'a self, table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(self.indexes.iter().filter(|i| i.table.eq_ignore_ascii_case(table)).collect())
        }
        fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
            Ok(())
        }
        fn create_table(&mut self, _p: &mut dyn Pager, _s: &CreateTable, _sql: &str) -> Result<()> {
            unreachable!("TestCatalog is read-only")
        }
        fn create_index(&mut self, _p: &mut dyn Pager, _s: &CreateIndex, _sql: &str) -> Result<()> {
            unreachable!("TestCatalog is read-only")
        }
        fn drop_object(&mut self, _p: &mut dyn Pager, _s: &Drop) -> Result<()> {
            unreachable!("TestCatalog is read-only")
        }
    }

    /// Assert one `index_info` row (`seqno`, `cid`, `name`) cell-by-cell. `name` is `Some`
    /// for a named column or `None` for an expression column's NULL name (`Value` has no
    /// `PartialEq`).
    fn assert_info_row(row: &[Value], seqno: i64, cid: i64, name: Option<&str>) {
        assert_eq!(row.len(), 3, "index_info row is 3 cells wide");
        assert!(matches!(&row[0], Value::Integer(v) if *v == seqno), "seqno: {:?}", row[0]);
        assert!(matches!(&row[1], Value::Integer(v) if *v == cid), "cid: {:?}", row[1]);
        match name {
            Some(n) => assert!(matches!(&row[2], Value::Text(t) if t == n), "name: {:?}", row[2]),
            None => assert!(matches!(&row[2], Value::Null), "name should be NULL: {:?}", row[2]),
        }
    }
}
