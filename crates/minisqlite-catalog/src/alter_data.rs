//! The `ALTER TABLE ... DROP COLUMN` row-data rewrite, kept apart from the schema
//! store so it lands in its own file and its streaming shape is unit-testable in
//! isolation. Removing a column shifts the stored ordinals of every column after it,
//! so each row's on-disk record must be rewritten to drop the value at that slot —
//! records are read positionally, so a stale value left at the wrong slot would
//! corrupt every later read.
//!
//! The `drop_idx` this rewrite removes is a PHYSICAL storage slot, NOT a schema column
//! ordinal. VIRTUAL generated columns take no storage slot (`gencol.html`), so the caller
//! (`SchemaCatalog::alter_drop_column`) maps the schema ordinal to a physical slot by
//! counting the non-virtual columns before it — and skips this rewrite entirely when the
//! dropped column is itself VIRTUAL (nothing is stored to remove). This file only ever
//! sees the already-mapped physical slot, so its `values.remove(drop_idx)` is correct.
//!
//! The rewrite is deliberately **streaming, one row resident at a time** (the
//! bounded-memory "stream, don't materialize" discipline): a single
//! `TableCursor` borrows the pager immutably, but the write-back needs `&mut pager`,
//! so we cannot hold the cursor open across the write. Instead we chain by rowid —
//! open a cursor, `seek_ge(next)` to the next unprocessed row, decode it, drop the
//! cursor (releasing the shared borrow), `table_insert` the narrowed record at the
//! SAME rowid (a REPLACE that strictly shrinks the record and preserves the rowid, so
//! every secondary index stays valid), then advance `next` past it and repeat. Peak
//! extra memory is one row (two reused buffers), independent of table size, versus the
//! O(table-data) cost of buffering every rewritten record first.

use minisqlite_btree::{table_insert, TableCursor};
use minisqlite_fileformat::{decode_record_into_enc, encode_record_into_enc, TextEncoding};
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Result, Value};

/// Rewrite every row of the rowid table rooted at `root_page`, removing the value at
/// PHYSICAL storage slot `drop_idx`, in O(1) extra memory (one row resident). `drop_idx`
/// is a stored-record slot (already virtual-adjusted by the caller), not a schema column
/// ordinal — see the module doc.
///
/// A "short" record from a prior `ADD COLUMN` may store fewer values than the table
/// has columns; when `drop_idx` is at or past a row's stored width there is nothing to
/// remove — the trailing columns were already implicitly NULL and stay correctly
/// trailing under the one-narrower schema, so the record is left untouched.
///
/// Must run inside the caller's write transaction so a mid-rewrite failure rolls back
/// with the schema change; the caller rewrites the `CREATE TABLE` text separately.
pub(crate) fn rewrite_drop_column_data(
    pager: &mut dyn Pager,
    root_page: PageId,
    drop_idx: usize,
    enc: TextEncoding,
) -> Result<()> {
    // Reused across every row so the hot loop makes no per-row allocation beyond what
    // a value's own owned bytes (Text/Blob) require.
    let mut values: Vec<Value> = Vec::new();
    let mut encoded: Vec<u8> = Vec::new();
    // rowids are signed; start below the minimum so `seek_ge` lands on the first row.
    let mut next: i64 = i64::MIN;
    loop {
        // Read exactly one row (>= `next`) into `values`, then drop the cursor so its
        // immutable pager borrow ends before the `&mut` write below.
        let rowid = {
            let mut cursor = TableCursor::open(&*pager, root_page)?;
            if !cursor.seek_ge(next)? {
                break;
            }
            let rowid = cursor.rowid();
            let payload = cursor.payload()?;
            // `decode_record_into_enc` clears `values` before appending, so the reused
            // buffer never carries a stale prefix from the previous row (the codec does for
            // `values` what the explicit `encoded.clear()` below does for `encoded`). TEXT
            // is decoded in the database's encoding (§1.3.13) and re-encoded in the same one
            // below, so a UTF-16 table's text survives the rewrite unchanged.
            decode_record_into_enc(&payload, enc, &mut values);
            rowid
        };

        if drop_idx < values.len() {
            values.remove(drop_idx);
        }
        // `encode_record_into_enc` APPENDS, so clear the reused buffer before each row —
        // otherwise every record would carry the previous row's bytes as a stale prefix.
        encoded.clear();
        encode_record_into_enc(&values, enc, &mut encoded);
        // REPLACE at the same rowid + root — preserves the key, so surviving indexes
        // (keyed by rowid) stay valid and `seek_ge(rowid + 1)` below still advances.
        table_insert(pager, root_page, rowid, &encoded)?;

        // A row at i64::MAX has no possible successor; stop rather than overflow.
        if rowid == i64::MAX {
            break;
        }
        next = rowid + 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_btree::create_table_btree;
    use minisqlite_fileformat::{decode_record, encode_record};
    use minisqlite_pager::MemPager;

    /// Build a fresh rowid b-tree and populate it with `(rowid, values)` rows.
    fn build_table(rows: &[(i64, Vec<Value>)]) -> (MemPager, PageId) {
        let mut pager = MemPager::new(4096);
        // Page 1 must exist as a valid header before allocating table roots.
        minisqlite_btree::init_database(&mut pager).unwrap();
        let root = create_table_btree(&mut pager).unwrap();
        for (rowid, vals) in rows {
            table_insert(&mut pager, root, *rowid, &encode_record(vals)).unwrap();
        }
        (pager, root)
    }

    /// Read back every row as `(rowid, values)` in ascending rowid order.
    fn dump(pager: &MemPager, root: PageId) -> Vec<(i64, Vec<Value>)> {
        let mut cursor = TableCursor::open(pager, root).unwrap();
        let mut out = Vec::new();
        let mut positioned = cursor.first().unwrap();
        while positioned {
            out.push((cursor.rowid(), decode_record(&cursor.payload().unwrap())));
            positioned = cursor.next().unwrap();
        }
        out
    }

    fn i(n: i64) -> Value {
        Value::Integer(n)
    }
    fn t(s: &str) -> Value {
        Value::Text(s.to_string())
    }

    /// `Value` is not `PartialEq` (an f64 lives inside), so rows are compared by their
    /// `Debug` form — the same convention the schemacatalog tests use for row data.
    fn assert_rows(got: Vec<(i64, Vec<Value>)>, want: Vec<(i64, Vec<Value>)>) {
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }

    #[test]
    fn drops_middle_column_from_every_row() {
        let (mut pager, root) = build_table(&[
            (1, vec![i(10), t("a"), i(100)]),
            (2, vec![i(20), t("b"), i(200)]),
        ]);
        rewrite_drop_column_data(&mut pager, root, 1, TextEncoding::Utf8).unwrap();
        assert_rows(
            dump(&pager, root),
            vec![(1, vec![i(10), i(100)]), (2, vec![i(20), i(200)])],
        );
    }

    #[test]
    fn drops_first_and_last_column() {
        let (mut pager, root) = build_table(&[(1, vec![i(10), t("a"), i(100)])]);
        rewrite_drop_column_data(&mut pager, root, 0, TextEncoding::Utf8).unwrap();
        assert_rows(dump(&pager, root), vec![(1, vec![t("a"), i(100)])]);

        let (mut pager, root) = build_table(&[(1, vec![i(10), t("a"), i(100)])]);
        rewrite_drop_column_data(&mut pager, root, 2, TextEncoding::Utf8).unwrap();
        assert_rows(dump(&pager, root), vec![(1, vec![i(10), t("a")])]);
    }

    #[test]
    fn short_row_shorter_than_drop_idx_is_untouched() {
        // A row storing only [10] under a 3-column schema (cols 1,2 implicitly NULL);
        // dropping ordinal 2 removes nothing, leaving the short record intact.
        let (mut pager, root) = build_table(&[(1, vec![i(10)]), (2, vec![i(20), t("b"), i(200)])]);
        rewrite_drop_column_data(&mut pager, root, 2, TextEncoding::Utf8).unwrap();
        assert_rows(
            dump(&pager, root),
            vec![(1, vec![i(10)]), (2, vec![i(20), t("b")])],
        );
    }

    #[test]
    fn empty_table_is_a_noop() {
        let (mut pager, root) = build_table(&[]);
        rewrite_drop_column_data(&mut pager, root, 0, TextEncoding::Utf8).unwrap();
        assert_rows(dump(&pager, root), vec![]);
    }

    #[test]
    fn negative_and_max_rowids_are_all_rewritten() {
        // Exercises the i64::MIN start (negative rowid) and the i64::MAX terminal break.
        let (mut pager, root) = build_table(&[
            (i64::MIN, vec![i(1), i(11)]),
            (-5, vec![i(2), i(22)]),
            (0, vec![i(3), i(33)]),
            (i64::MAX, vec![i(4), i(44)]),
        ]);
        rewrite_drop_column_data(&mut pager, root, 0, TextEncoding::Utf8).unwrap();
        assert_rows(
            dump(&pager, root),
            vec![
                (i64::MIN, vec![i(11)]),
                (-5, vec![i(22)]),
                (0, vec![i(33)]),
                (i64::MAX, vec![i(44)]),
            ],
        );
    }

    #[test]
    fn many_rows_stream_correctly() {
        // A wider run to exercise multi-leaf seek_ge chaining (each row re-descends).
        let rows: Vec<(i64, Vec<Value>)> =
            (1..=500).map(|n| (n, vec![i(n), t("payload-to-drop"), i(n * 2)])).collect();
        let (mut pager, root) = build_table(&rows);
        rewrite_drop_column_data(&mut pager, root, 1, TextEncoding::Utf8).unwrap();
        let want: Vec<(i64, Vec<Value>)> = (1..=500).map(|n| (n, vec![i(n), i(n * 2)])).collect();
        assert_rows(dump(&pager, root), want);
    }
}
