//! AUTOINCREMENT rowid bookkeeping over the internal `sqlite_sequence(name, seq)` table
//! (`spec/sqlite-doc/autoinc.html` §2–§3).
//!
//! An `INTEGER PRIMARY KEY AUTOINCREMENT` column must generate rowids that are monotonic
//! AND never reused, even after the largest row — or every row — is deleted. A plain
//! `INTEGER PRIMARY KEY` seeds the next rowid from `max(rowid) + 1` and so REUSES a rowid
//! freed by a delete; AUTOINCREMENT instead records the largest rowid EVER used for the
//! table in `sqlite_sequence.seq` and seeds from `max(max(rowid), seq) + 1`, so a freed
//! rowid is never handed out again.
//!
//! Division of labor (per the engine's design): the catalog auto-creates the
//! `sqlite_sequence(name, seq)` table when a table declaring AUTOINCREMENT is created
//! (`schemacatalog::ensure_sqlite_sequence`), but it does NOT insert a per-table
//! bookkeeping row — real sqlite writes that row lazily on the first insert, so
//! [`read_sequence`] returns `None` until then (a conformance test pins that
//! `sqlite_sequence` is empty before the first insert). The INSERT operator owns reading
//! the high-water before it assigns rowids and writing it back after — that "read+write
//! the row for this table" split is exactly this module.
//!
//! WIRING: `insert.rs::build` consumes these — it seeds `next_auto` from
//! `max(table_max, read_sequence(..).seq)` when `def.autoincrement`, advances it per
//! written row (explicit or generated), and calls `write_sequence(.., next_auto)` once
//! after the write loop (gated on a row actually being inserted). The `autoincrement`
//! schema flag is the trigger because a plain vs. an AUTOINCREMENT `INTEGER PRIMARY KEY`
//! is indistinguishable at the row level (the first insert of each looks identical and
//! neither has a seq row yet), so the presence of a seq row cannot be the signal.

use minisqlite_btree::{table_insert, TableCursor};
use minisqlite_catalog::Catalog;
use minisqlite_fileformat::encode_record_enc;
use minisqlite_pager::{text_encoding_of, Pager};
use minisqlite_types::{DbIndex, Result, Value};

use crate::row::decode_table_row_enc;

/// The internal bookkeeping table (`autoinc.html` §3). Its column order is `(name, seq)`.
const SEQUENCE_TABLE: &str = "sqlite_sequence";

/// One `sqlite_sequence` bookkeeping row for a single AUTOINCREMENT table: the b-tree
/// `rowid` of that row (so a writer can rewrite it in place) and the `seq` high-water it
/// records (the largest rowid ever used for the table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SequenceRow {
    pub rowid: i64,
    pub seq: i64,
}

/// Read the `seq` high-water `sqlite_sequence` records for `table`, or `None` when there
/// is no row for it yet (its first AUTOINCREMENT insert) — or when `sqlite_sequence` does
/// not exist at all (no AUTOINCREMENT table has been created). A row whose `seq` is NULL or
/// a non-integer counts as "no high-water" (`seq == 0`), matching a just-materialized row.
pub(crate) fn read_sequence(
    pager: &dyn Pager,
    catalog: &dyn Catalog,
    db: DbIndex,
    table: &str,
) -> Result<Option<SequenceRow>> {
    // `sqlite_sequence` is PER-database (each store has its own, autoinc.html §3): `db` is the
    // write's namespace and `pager` is already that store's, so read the `sqlite_sequence` DEF
    // from the SAME store via `table_in(db)`. The bare `table` follows the temp->main->attached
    // search order and, when another namespace also has AUTOINCREMENT tables, would return that
    // store's `sqlite_sequence` — whose `root_page` would then be opened on THIS store's pager,
    // reading an unrelated page. With no such shadow `db == MAIN` and this equals the bare form.
    let Some(seq_def) = catalog.table_in(db, SEQUENCE_TABLE)? else {
        return Ok(None);
    };
    // `sqlite_sequence.name` is TEXT; on a UTF-16 database it is stored UTF-16 (write_sequence
    // encodes it that way), so decode it in the DB's encoding — a bare UTF-8 decode would read
    // `t\0…` and never match `table`, losing the AUTOINCREMENT high-water. Read once, before the
    // scan loop, so the UTF-8 path pays no per-row header parse.
    let enc = text_encoding_of(pager);
    let mut tc = TableCursor::open(pager, seq_def.root_page)?;
    if !tc.first()? {
        return Ok(None);
    }
    loop {
        let rowid = tc.rowid();
        // Scope the payload borrow closed before `tc.next()` reborrows the cursor mutably.
        let found = {
            let payload = tc.payload()?;
            let row = decode_table_row_enc(&payload, rowid, seq_def, enc);
            match row.first() {
                Some(Value::Text(name)) if name == table => Some(match row.get(1) {
                    Some(Value::Integer(i)) => *i,
                    _ => 0,
                }),
                _ => None,
            }
        };
        if let Some(seq) = found {
            return Ok(Some(SequenceRow { rowid, seq }));
        }
        if !tc.next()? {
            return Ok(None);
        }
    }
}

/// Upsert the `seq` high-water for `table` to `new_seq` in `sqlite_sequence`. When a
/// bookkeeping row already exists it is rewritten IN PLACE at the same b-tree rowid (a
/// `table_insert` at an existing key replaces its payload), so the row is never
/// duplicated; otherwise a new row is appended at `sqlite_sequence`'s own `max(rowid) + 1`.
/// A no-op when `sqlite_sequence` does not exist (the caller only invokes this for an
/// AUTOINCREMENT table, so that branch is purely defensive).
pub(crate) fn write_sequence(
    pager: &mut dyn Pager,
    catalog: &dyn Catalog,
    db: DbIndex,
    table: &str,
    new_seq: i64,
) -> Result<()> {
    // Same per-database scoping as `read_sequence`: resolve `sqlite_sequence` in the write's own
    // namespace `db` (matching `pager`), never the search-order union.
    let Some(seq_def) = catalog.table_in(db, SEQUENCE_TABLE)? else {
        return Ok(());
    };
    let root = seq_def.root_page;
    // Locate the existing bookkeeping row (if any) via a shared scan, returned owned so its
    // borrow ends before the exclusive write below (the operator never holds both at once).
    let existing = read_sequence(&*pager, catalog, db, table)?;
    // `sqlite_sequence.name` is TEXT, so store it in the database's encoding (UTF-16 for a
    // UTF-16 database) — same as every other data row — so real sqlite reads it back.
    let enc = text_encoding_of(&*pager);
    let record =
        encode_record_enc(&[Value::Text(table.to_string()), Value::Integer(new_seq)], enc);
    let target_rowid = match existing {
        Some(SequenceRow { rowid, .. }) => rowid,
        None => {
            let mut tc = TableCursor::open(&*pager, root)?;
            if tc.last()? {
                tc.rowid() + 1
            } else {
                1
            }
        }
    };
    table_insert(pager, root, target_rowid, &record)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_btree::init_database;
    use minisqlite_catalog::SchemaCatalog;
    use minisqlite_pager::MemPager;
    use minisqlite_sql::{parse, Statement};

    /// A fresh in-memory database with `create_sql` applied through the real catalog path
    /// (so an AUTOINCREMENT table auto-creates `sqlite_sequence` exactly as the engine does).
    fn db(create_sql: &str) -> (MemPager, SchemaCatalog) {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let mut cat = SchemaCatalog::new();
        create(&mut pager, &mut cat, create_sql);
        (pager, cat)
    }

    fn create(pager: &mut MemPager, cat: &mut SchemaCatalog, create_sql: &str) {
        let ast = parse(create_sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else {
            panic!("not a CREATE TABLE: {create_sql}");
        };
        cat.create_table(pager, stmt, create_sql).unwrap();
    }

    #[test]
    fn plain_table_has_no_sequence_table_and_reads_none() {
        // A plain INTEGER PRIMARY KEY (no AUTOINCREMENT) creates no sqlite_sequence.
        let (pager, cat) = db("CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
        assert!(cat.table("sqlite_sequence").unwrap().is_none());
        assert_eq!(read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap(), None);
    }

    #[test]
    fn autoincrement_table_creates_sequence_but_no_row_until_first_write() {
        // sqlite creates sqlite_sequence at CREATE TABLE but writes no row until the first
        // insert — read_sequence must report None so the operator seeds from the table max.
        let (pager, cat) = db("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
        assert!(cat.table("sqlite_sequence").unwrap().is_some());
        assert_eq!(read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap(), None);
    }

    #[test]
    fn write_then_read_round_trips() {
        let (mut pager, cat) = db("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
        write_sequence(&mut pager, &cat, DbIndex::MAIN, "t", 5).unwrap();
        let sr = read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap().unwrap();
        assert_eq!(sr.seq, 5);
    }

    #[test]
    fn second_write_updates_in_place_never_duplicates_the_row() {
        let (mut pager, cat) = db("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
        write_sequence(&mut pager, &cat, DbIndex::MAIN, "t", 5).unwrap();
        let first = read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap().unwrap();
        write_sequence(&mut pager, &cat, DbIndex::MAIN, "t", 9).unwrap();
        let second = read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap().unwrap();
        // If the update appended a second row, the ascending scan would still return the
        // ORIGINAL (seq 5) at the lower rowid — so seeing 9 at the SAME rowid proves the
        // row was rewritten in place, not duplicated.
        assert_eq!(second.seq, 9);
        assert_eq!(second.rowid, first.rowid, "seq row rewritten in place, not duplicated");
    }

    #[test]
    fn two_autoincrement_tables_track_independent_high_waters() {
        let (mut pager, mut cat) = db("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
        create(&mut pager, &mut cat, "CREATE TABLE u(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
        write_sequence(&mut pager, &cat, DbIndex::MAIN, "t", 3).unwrap();
        write_sequence(&mut pager, &cat, DbIndex::MAIN, "u", 100).unwrap();
        assert_eq!(read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap().unwrap().seq, 3);
        assert_eq!(read_sequence(&pager, &cat, DbIndex::MAIN, "u").unwrap().unwrap().seq, 100);
        // Updating one leaves the other untouched.
        write_sequence(&mut pager, &cat, DbIndex::MAIN, "t", 7).unwrap();
        assert_eq!(read_sequence(&pager, &cat, DbIndex::MAIN, "t").unwrap().unwrap().seq, 7);
        assert_eq!(read_sequence(&pager, &cat, DbIndex::MAIN, "u").unwrap().unwrap().seq, 100);
    }
}
