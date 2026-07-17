//! Conformance battery: **GENERATED COLUMNS** — both VIRTUAL (computed on read, not
//! stored) and STORED (computed on write, physically stored).
//!
//! Every expectation is TRANSCRIBED FROM THE SPEC (`spec/sqlite-doc/gencol.html`,
//! `pragma.html`), never from what the engine currently returns — a failing case is
//! the intended signal that the engine diverges from the spec.
//!
//! Spec anchors (`spec/sqlite-doc/gencol.html`):
//!   * "If the trailing VIRTUAL or STORED keyword is omitted, then VIRTUAL is the
//!     default." — so a bare `x AS (expr)` is VIRTUAL (the common case).
//!   * "The value of a VIRTUAL column is computed when read, whereas the value of a
//!     STORED column is computed when the row is written. STORED columns take up space
//!     in the database file, whereas VIRTUAL columns use more CPU cycles when read." —
//!     the on-disk invariant: VIRTUAL is not stored, STORED is.
//!   * "Generated columns can be read, but their values can not be directly written." —
//!     the INSERT/UPDATE reject.
//!   * "The expression of a generated column ... can reference the INTEGER PRIMARY KEY
//!     column."
//!   * "generated columns ... are not shown by PRAGMA table_info ... they are included
//!     in the output of the newer PRAGMA table_xinfo statement." (+ `pragma.html`
//!     #pragma_table_xinfo: hidden = 0 normal, 2 VIRTUAL, 3 STORED.)
//!
//! The exact reject error text matches real sqlite: `cannot INSERT into generated
//! column "<name>"` and `cannot UPDATE generated column "<name>"`.

mod conformance;

use conformance::*;

use minisqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ===========================================================================
// VIRTUAL (the default) — computed on read, available to every operator.
// ===========================================================================

#[test]
fn virtual_default_computes_on_read() {
    // gencol.html: a bare `AS (expr)` with no keyword is VIRTUAL, computed when read.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    // The generated column read on its own, and alongside the base column.
    assert_scalar(&mut db, "SELECT b FROM t", int(6));
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(5), int(6)]]);
}

#[test]
fn virtual_uses_other_column_and_a_function() {
    // gencol.html: the expression is "a function of other columns in the same row" and
    // may call deterministic scalar functions.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x TEXT, y AS (upper(x)))");
    exec(&mut db, "INSERT INTO t(x) VALUES ('abc')");
    assert_scalar(&mut db, "SELECT y FROM t", text("ABC"));
}

#[test]
fn virtual_references_earlier_generated_column() {
    // gencol.html: a generated column may reference another generated column defined
    // EARLIER (the dependency is already satisfied; no cycle). The forward-reference case
    // (a column referencing one declared LATER) is exercised by
    // `generated_column_can_reference_a_later_generated_column` below.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1), c AS (b + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    assert_rows(&mut db, "SELECT a, b, c FROM t", &[vec![int(5), int(6), int(7)]]);
}

#[test]
fn generated_references_integer_primary_key() {
    // gencol.html: the expression "can reference the INTEGER PRIMARY KEY column". The
    // rowid it aliases must be visible to the generation expression — including when the
    // rowid is AUTO-assigned (a NULL id), which fixes the value only during the write.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, b AS (id * 10))");
    exec(&mut db, "INSERT INTO t(id) VALUES (3)");
    exec(&mut db, "INSERT INTO t(id) VALUES (NULL)"); // auto rowid = 4
    assert_rows(
        &mut db,
        "SELECT id, b FROM t ORDER BY id",
        &[vec![int(3), int(30)], vec![int(4), int(40)]],
    );
}

// ===========================================================================
// The generated value is available beyond projection: WHERE / ORDER BY /
// aggregates / DELETE all see it.
// ===========================================================================

#[test]
fn where_filters_on_generated_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5), (1), (3)");
    // b = 6, 2, 4 → the row with b = 4 has a = 3.
    assert_scalar(&mut db, "SELECT a FROM t WHERE b = 4", int(3));
}

#[test]
fn order_by_on_generated_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5), (1), (3)");
    assert_rows(
        &mut db,
        "SELECT b FROM t ORDER BY b",
        &[vec![int(2)], vec![int(4)], vec![int(6)]],
    );
}

#[test]
fn aggregate_over_generated_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a * 2))");
    exec(&mut db, "INSERT INTO t(a) VALUES (1), (2), (3)");
    // b = 2, 4, 6 → SUM(b) = 12.
    assert_scalar(&mut db, "SELECT sum(b) FROM t", int(12));
}

#[test]
fn delete_where_on_generated_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5), (1), (3)");
    // Delete the row whose VIRTUAL b = 2 (a = 1); the others survive.
    exec(&mut db, "DELETE FROM t WHERE b = 2");
    assert_rows(
        &mut db,
        "SELECT a FROM t ORDER BY a",
        &[vec![int(3)], vec![int(5)]],
    );
}

// ===========================================================================
// Affinity — the computed value is coerced to the column's declared type.
// ===========================================================================

#[test]
fn generated_value_takes_column_affinity() {
    // A generated column's declared type gives it affinity, applied to the computed
    // value like any ordinary column: a TEXT-affinity column turns the integer `5` into
    // the text `'5'` (types::apply_affinity), so the class is TEXT, not INTEGER.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b TEXT AS (a))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    assert_scalar(&mut db, "SELECT b FROM t", text("5"));
    assert_scalar(&mut db, "SELECT typeof(b) FROM t", text("text"));
}

// ===========================================================================
// Write-path rejects: a generated column may not be assigned directly.
// ===========================================================================

#[test]
fn reject_insert_into_generated_column() {
    // gencol.html: "their values can not be directly written". Naming a generated column
    // in an INSERT target list is real sqlite's `cannot INSERT into generated column
    // "<name>"`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    let e = assert_exec_error(&mut db, "INSERT INTO t(a, b) VALUES (1, 2)");
    assert!(
        e.to_string().contains(r#"cannot INSERT into generated column "b""#),
        "expected the exact sqlite reject text, got: {e}"
    );
}

#[test]
fn reject_update_generated_column() {
    // gencol.html: the only way to change a generated column is to change its inputs.
    // Assigning it in UPDATE is real sqlite's `cannot UPDATE generated column "<name>"`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (1)");
    let e = assert_exec_error(&mut db, "UPDATE t SET b = 5");
    assert!(
        e.to_string().contains(r#"cannot UPDATE generated column "b""#),
        "expected the exact sqlite reject text, got: {e}"
    );
}

#[test]
fn positional_insert_excludes_generated_column() {
    // An all-columns (positional) INSERT supplies values for the NON-generated columns
    // only — a generated column is never user-supplied. `INSERT INTO t VALUES (5)` must
    // therefore set a = 5 (not error "2 columns but 1 value") and compute b.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t VALUES (5)");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(5), int(6)]]);
}

#[test]
fn positional_insert_excludes_generated_column_multi_row() {
    // Same rule across a multi-row VALUES list and with the generated column in the
    // MIDDLE of the table (proving the exclusion is positional-aware, not "trailing").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, g AS (a * 2), c INT)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20)");
    assert_rows(
        &mut db,
        "SELECT a, g, c FROM t ORDER BY a",
        &[vec![int(1), int(2), int(10)], vec![int(2), int(4), int(20)]],
    );
}

// ===========================================================================
// UPDATE of a base column recomputes the generated column(s).
// ===========================================================================

#[test]
fn update_base_column_recomputes_generated() {
    // gencol.html: "The only way to change the value of a generated column is to modify
    // the values of the other columns used to calculate" it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    exec(&mut db, "UPDATE t SET a = 10");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(10), int(11)]]);
}

#[test]
fn generated_column_can_reference_a_later_generated_column() {
    // gencol.html §2.2: "Generated columns can occur anywhere in the table definition ...
    // interspersed among ordinary columns." and "The expression of a generated column can
    // refer to any of the other declared columns in the table, including other generated
    // columns, as long as [it does] not directly or indirectly refer back to itself." So a
    // generated column may reference one declared LATER — evaluation must follow the
    // dependency order, not the textual column order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (c + 1), c AS (a * 2))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    // c = a*2 = 10; b = c+1 = 11.
    assert_rows(&mut db, "SELECT a, b, c FROM t", &[vec![int(5), int(11), int(10)]]);
}

#[test]
fn self_referential_generated_column_is_rejected_not_looping() {
    // gencol.html §2.2: "no generated column can depend upon itself, either directly or
    // indirectly." A direct self-reference must be a loud error (real SQLite rejects such a
    // schema), never an infinite loop or a silently NULL-fed value. The error may surface at
    // CREATE TABLE or at first use of the table; either is acceptable — the invariant under
    // test is that the engine errors and does NOT hang or compute a wrong value.
    let mut db = mem();
    if try_exec(&mut db, "CREATE TABLE t(a INT, b AS (b + 1))").is_ok() {
        assert!(
            try_exec(&mut db, "INSERT INTO t(a) VALUES (1)").is_err(),
            "a self-referential generated column must error, not compute a value"
        );
    }
}

#[test]
fn cyclic_generated_columns_are_rejected_not_looping() {
    // gencol.html §2.2: an INDIRECT cycle (b→c→b) is likewise forbidden. Must error, never
    // loop. Surfacing at CREATE or first use are both acceptable.
    let mut db = mem();
    if try_exec(&mut db, "CREATE TABLE t(a INT, b AS (c + 1), c AS (b + 1))").is_ok() {
        assert!(
            try_exec(&mut db, "INSERT INTO t(a) VALUES (1)").is_err(),
            "cyclic generated columns must error, not loop or compute a wrong value"
        );
    }
}

#[test]
fn insert_returning_sees_computed_generated_column() {
    // A generated column is computed BEFORE RETURNING evaluates, so `RETURNING b` reports
    // the computed value (the generated value is part of the row RETURNING sees).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1), s INT AS (a * 2) STORED)");
    assert_rows(
        &mut db,
        "INSERT INTO t(a) VALUES (5) RETURNING a, b, s",
        &[vec![int(5), int(6), int(10)]],
    );
}

#[test]
fn update_recomputes_stored_generated_column() {
    // A STORED generated column is recomputed on the write, so an UPDATE of its input
    // rewrites the stored value too.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, s INT AS (a * 2) STORED)");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    assert_scalar(&mut db, "SELECT s FROM t", int(10));
    exec(&mut db, "UPDATE t SET a = 7");
    assert_scalar(&mut db, "SELECT s FROM t", int(14));
}

// ===========================================================================
// Indexes on generated columns.
// ===========================================================================

#[test]
fn index_on_virtual_generated_column_returns_correct_rows() {
    // An index built over a VIRTUAL column indexes its COMPUTED value; a lookup on it
    // returns the correct row. (Backfill computes the value at CREATE INDEX time, and
    // later INSERT/UPDATE maintain the entry from the computed row.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (1), (2), (3)");
    exec(&mut db, "CREATE INDEX ix ON t(b)");
    assert_scalar(&mut db, "SELECT a FROM t WHERE b = 3", int(2));
    // A later insert must maintain the index: b = 5 (a = 4) becomes findable.
    exec(&mut db, "INSERT INTO t(a) VALUES (4)");
    assert_scalar(&mut db, "SELECT a FROM t WHERE b = 5", int(4));
    // An update that moves the input moves the index entry: a=2 → b was 3, now 3→ still
    // find via new value. Move a=1 (b=2) to a=9 (b=10) and check both keys.
    exec(&mut db, "UPDATE t SET a = 9 WHERE a = 1");
    assert_rows(&mut db, "SELECT a FROM t WHERE b = 2", &[]);
    assert_scalar(&mut db, "SELECT a FROM t WHERE b = 10", int(9));
}

#[test]
fn index_on_stored_generated_column_returns_correct_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, s INT AS (a * 2) STORED)");
    exec(&mut db, "INSERT INTO t(a) VALUES (1), (2), (3)");
    exec(&mut db, "CREATE INDEX ix ON t(s)");
    assert_scalar(&mut db, "SELECT a FROM t WHERE s = 4", int(2));
}

#[test]
fn delete_removes_index_entry_on_virtual_column() {
    // A DELETE must remove the deleted row's index entry keyed on the COMPUTED virtual
    // value (the scan supplies the computed row to the index-delete). After deleting the
    // row whose b = 3, a lookup on b = 3 finds nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (1), (2), (3)");
    exec(&mut db, "CREATE INDEX ix ON t(b)");
    exec(&mut db, "DELETE FROM t WHERE a = 2"); // b = 3
    assert_rows(&mut db, "SELECT a FROM t WHERE b = 3", &[]);
    assert_rows(&mut db, "SELECT a FROM t ORDER BY a", &[vec![int(1)], vec![int(3)]]);
}

// ===========================================================================
// WITHOUT ROWID tables — a generated column is stored (STORED) / recomputed
// (VIRTUAL) with the PK-keyed record layout, never taking a storage slot when
// VIRTUAL (the WR record omits it, just like a rowid table).
// ===========================================================================

#[test]
fn without_rowid_virtual_generated_computes_on_read() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT PRIMARY KEY, b AS (a + 1)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t(a) VALUES (5), (1)");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), int(2)], vec![int(5), int(6)]],
    );
}

#[test]
fn without_rowid_stored_generated_persists_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(
            &mut db,
            "CREATE TABLE t(a INT PRIMARY KEY, s INT AS (a * 2) STORED, v INT AS (a + 1)) \
             WITHOUT ROWID",
        );
        exec(&mut db, "INSERT INTO t(a) VALUES (5)");
        assert_rows(&mut db, "SELECT a, s, v FROM t", &[vec![int(5), int(10), int(6)]]);
    }
    {
        let mut db = tmp.open();
        assert_rows(&mut db, "SELECT a, s, v FROM t", &[vec![int(5), int(10), int(6)]]);
    }
}

// ===========================================================================
// PRAGMA table_xinfo / table_info.
// ===========================================================================

#[test]
fn table_xinfo_reports_hidden_flags() {
    // pragma.html #pragma_table_xinfo: hidden = 0 for a normal column, 2 for a VIRTUAL
    // generated column, 3 for a STORED generated column. Every column is listed with its
    // TRUE 0-based cid.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE t(a INTEGER, b INT AS (a + 1) VIRTUAL, c INT AS (a * 2) STORED, d INT)",
    );
    assert_columns(
        &mut db,
        "PRAGMA table_xinfo(t)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk", "hidden"],
    );
    assert_rows(
        &mut db,
        "PRAGMA table_xinfo(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0), int(0)],
            vec![int(1), text("b"), text("INT"), int(0), null(), int(0), int(2)],
            vec![int(2), text("c"), text("INT"), int(0), null(), int(0), int(3)],
            vec![int(3), text("d"), text("INT"), int(0), null(), int(0), int(0)],
        ],
    );
}

#[test]
fn table_info_excludes_generated_and_renumbers_cid() {
    // gencol.html / pragma.html: table_info does NOT show generated columns. `cid` is
    // "rank within the current result set", so the emitted columns are 0,1,2,…
    // contiguously — here a (cid 0) and d (cid 1), with the omitted b/c leaving NO gap.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE t(a INTEGER, b INT AS (a + 1) VIRTUAL, c INT AS (a * 2) STORED, d INT)",
    );
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0)],
            vec![int(1), text("d"), text("INT"), int(0), null(), int(0)],
        ],
    );
}

// ===========================================================================
// On-disk / durability: STORED persists across a reopen; VIRTUAL recomputes.
// ===========================================================================

#[test]
fn stored_persists_and_virtual_recomputes_across_reopen() {
    // gencol.html on-disk invariant: a STORED column's value is written to the record
    // (present, byte-for-byte, after a reopen); a VIRTUAL column is NOT stored, so after
    // a reopen it is recomputed from the base column. Both must read back correctly.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(
            &mut db,
            "CREATE TABLE t(a INT, s INT AS (a * 2) STORED, v INT AS (a + 1) VIRTUAL)",
        );
        exec(&mut db, "INSERT INTO t(a) VALUES (5)");
        assert_rows(
            &mut db,
            "SELECT a, s, v FROM t",
            &[vec![int(5), int(10), int(6)]],
        );
    }
    // Reopen from the file: the STORED value comes from the record, the VIRTUAL value is
    // recomputed on the scan — both must still be exact.
    {
        let mut db = tmp.open();
        assert_rows(
            &mut db,
            "SELECT a, s, v FROM t",
            &[vec![int(5), int(10), int(6)]],
        );
        // And an UPDATE after reopen still recomputes both.
        exec(&mut db, "UPDATE t SET a = 8");
        assert_rows(
            &mut db,
            "SELECT a, s, v FROM t",
            &[vec![int(8), int(16), int(9)]],
        );
    }
}

#[test]
fn stored_generated_depending_on_a_later_stored_generated_persists_across_reopen() {
    // Write-path dependency ordering for STORED columns, pinned across a reopen. `s2` is a
    // STORED column that depends on ANOTHER generated column (`s1`) which is declared LATER,
    // so the toposort must compute s1 before s2 at WRITE time and store BOTH values. A reopen
    // reads them straight from the record (no read-recompute can mask a mis-ordered or stale
    // stored value the way an all-VIRTUAL chain would), so this is the one dependency-ordering
    // corner an all-virtual reverse-chain test cannot cover. a=4 => s1=40, s2=41.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(
            &mut db,
            "CREATE TABLE t(a INT, s2 AS (s1 + 1) STORED, s1 AS (a * 10) STORED)",
        );
        exec(&mut db, "INSERT INTO t(a) VALUES (4)");
        assert_rows(
            &mut db,
            "SELECT a, s1, s2 FROM t",
            &[vec![int(4), int(40), int(41)]],
        );
    }
    // Reopen: both STORED values must come back from disk exactly (no recompute path here).
    {
        let mut db = tmp.open();
        assert_rows(
            &mut db,
            "SELECT a, s1, s2 FROM t",
            &[vec![int(4), int(40), int(41)]],
        );
        // An UPDATE after reopen re-derives the chain in order and re-stores both.
        exec(&mut db, "UPDATE t SET a = 7");
        assert_rows(
            &mut db,
            "SELECT a, s1, s2 FROM t",
            &[vec![int(7), int(70), int(71)]],
        );
    }
}

// ===========================================================================
// Constraints and non-seq-scan read paths over a generated value. gencol.html
// §2.2/§2.3: a generated column participates in CHECK/NOT NULL, and its value is
// computed on every read path (a rowid point lookup here, not just a seq scan)
// and on every write path (an UPSERT DO UPDATE rewrite).
// ===========================================================================

#[test]
fn check_constraint_sees_computed_generated_value() {
    // gencol.html §2.3: "The expression of a CHECK constraint may ... reference generated
    // columns." The CHECK runs against the COMPUTED value, so an insert whose base column
    // drives the generated value out of range is rejected; an in-range one is kept.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b AS (a + 1), CHECK (b < 100))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)"); // b = 6 < 100 -> OK
    assert_scalar(&mut db, "SELECT b FROM t", int(6));
    // b = 200 violates CHECK(b < 100) -> the row is rejected, not stored.
    assert!(try_exec(&mut db, "INSERT INTO t(a) VALUES (199)").is_err());
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

#[test]
fn rowid_point_lookup_computes_virtual_generated_column() {
    // A WHERE on the INTEGER PRIMARY KEY is a rowid point lookup (RowidScan) — a different
    // leaf from the seq/index scans the other tests cover — and the VIRTUAL column must
    // still be computed there. gencol.html: the expression may reference the INTEGER
    // PRIMARY KEY column.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a INT, v AS (a + id))");
    exec(&mut db, "INSERT INTO t(id, a) VALUES (3, 10)");
    exec(&mut db, "INSERT INTO t(id, a) VALUES (7, 100)");
    assert_scalar(&mut db, "SELECT v FROM t WHERE id = 3", int(13));
    assert_scalar(&mut db, "SELECT v FROM t WHERE id = 7", int(107));
}

#[test]
fn upsert_do_update_recomputes_generated_column() {
    // ON CONFLICT ... DO UPDATE rewrites the row, so its generated columns must be
    // recomputed from the updated base column (gencol.html: the value is computed when the
    // row is written). A DO UPDATE that changes the base must not leave the generated value
    // stale.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INT, b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(k, a) VALUES (1, 10)");
    assert_scalar(&mut db, "SELECT b FROM t", int(11));
    exec(
        &mut db,
        "INSERT INTO t(k, a) VALUES (1, 99) ON CONFLICT(k) DO UPDATE SET a = excluded.a",
    );
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(99), int(100)]]);
}

// ===========================================================================
// FOREIGN KEY + generated column. gencol.html §2.2 lists FOREIGN KEY among the
// constraints a generated column participates in, and a table with a VIRTUAL
// generated column stores a NARROWER record (the virtual column is omitted), so
// the FK parent/child scans must decode virtual-aware — a positional decode
// shifts every column after the virtual one and reads the FK key column wrong
// (typically NULL), producing a spurious violation or a missed cascade.
// ===========================================================================

#[test]
fn fk_parent_scan_reads_ordinary_key_after_a_virtual_column() {
    // Parent `p` has a VIRTUAL column `b` at index 1 and the referenced UNIQUE key `c` at
    // index 2. The stored record omits `b`, so it is [a, c]; a positional decode would put
    // `c` in `b`'s slot and read the key as NULL. The child insert references an EXISTING
    // parent key, so real sqlite ACCEPTS it.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys=ON");
    exec(&mut db, "CREATE TABLE p(a INT, b AS (a + 100), c INT UNIQUE)");
    exec(&mut db, "INSERT INTO p(a, c) VALUES (1, 50)");
    assert_rows(&mut db, "SELECT a, b, c FROM p", &[vec![int(1), int(101), int(50)]]);
    exec(&mut db, "CREATE TABLE ch(x INT REFERENCES p(c))");
    exec(&mut db, "INSERT INTO ch(x) VALUES (50)"); // p.c=50 exists -> accepted
    assert_scalar(&mut db, "SELECT x FROM ch", int(50));
}

#[test]
fn fk_child_check_with_virtual_column_before_the_key() {
    // The virtual column precedes the key in the PARENT: `p(g VIRTUAL, k UNIQUE)`. Stored
    // record is [k]; a positional decode reads k into g's slot and leaves k NULL, so the
    // existing parent row is reported missing and a valid child insert is wrongly rejected.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys=ON");
    exec(&mut db, "CREATE TABLE p(g AS (k + 1), k INTEGER UNIQUE)");
    exec(&mut db, "CREATE TABLE c(fk INTEGER REFERENCES p(k))");
    exec(&mut db, "INSERT INTO p(k) VALUES (5)");
    exec(&mut db, "INSERT INTO c(fk) VALUES (5)"); // p.k=5 exists -> accepted
    assert_scalar(&mut db, "SELECT fk FROM c", int(5));
    // A child value with NO matching parent is still correctly rejected (not masked).
    assert!(try_exec(&mut db, "INSERT INTO c(fk) VALUES (99)").is_err());
}

#[test]
fn fk_cascade_delete_reaches_child_with_a_virtual_column() {
    // The child `c` has a VIRTUAL column `g` before the FK column `ref`. On parent delete,
    // matching_child_rows must decode `c` virtual-aware to find `ref`; a positional decode
    // reads `ref` as NULL, matches no child, and the CASCADE deletes nothing (orphan).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys=ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(g AS (ref + 1), ref INTEGER REFERENCES p(id) ON DELETE CASCADE)");
    exec(&mut db, "INSERT INTO p VALUES (1)");
    exec(&mut db, "INSERT INTO c(ref) VALUES (1)");
    assert_rows(&mut db, "SELECT ref, g FROM c", &[vec![int(1), int(2)]]);
    exec(&mut db, "DELETE FROM p WHERE id = 1"); // must cascade-delete c's row
    assert_rows(&mut db, "SELECT ref FROM c", &[]);
}

#[test]
fn fk_references_a_virtual_generated_parent_column() {
    // The referenced PARENT key is ITSELF a VIRTUAL generated column (UNIQUE). It is not in
    // the stored record, so the FK check must COMPUTE it to compare — a virtual-aware decode
    // alone leaves it NULL. p has u = k+1 (VIRTUAL UNIQUE); child references p(u).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys=ON");
    exec(&mut db, "CREATE TABLE p(k INT, u AS (k + 1) UNIQUE)");
    exec(&mut db, "INSERT INTO p(k) VALUES (5)"); // u = 6
    exec(&mut db, "CREATE TABLE c(fk INT REFERENCES p(u))");
    exec(&mut db, "INSERT INTO c(fk) VALUES (6)"); // p.u=6 exists -> accepted
    assert_scalar(&mut db, "SELECT fk FROM c", int(6));
    assert!(try_exec(&mut db, "INSERT INTO c(fk) VALUES (7)").is_err()); // no p.u=7
}

// ===========================================================================
// RECURSIVE TRIGGERS + generated columns. A trigger action reached at recursion
// depth >= 2 runs a RECOMPILED trigger set (PRAGMA recursive_triggers = ON): the
// compile pass expands only one trigger level, so a deeper action is compiled at
// RUNTIME and must ALSO have its generated-column programs bound. Otherwise the
// deep write runs with an empty program map — it PHYSICALLY STORES a VIRTUAL
// column (an on-disk-format corruption) and leaves its computed value NULL.
// gencol.html: a VIRTUAL column is never stored; its value is computed on read.
// ===========================================================================

#[test]
fn recursive_trigger_action_computes_generated_columns() {
    // The n=3 row is inserted by the depth-2 action, compiled by the runtime trigger
    // recompile path. Its VIRTUAL column g must be computed on read and NOT stored, and
    // the ordinary column m (after g in the schema) must read back its DEFAULT 7 — a
    // stored VIRTUAL slot would shift m to NULL on the virtual-skip decode.
    let mut db = mem();
    exec(&mut db, "PRAGMA recursive_triggers = ON");
    exec(&mut db, "CREATE TABLE c(n INTEGER, g INTEGER AS (n + 100) VIRTUAL, m INTEGER DEFAULT 7)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON c WHEN NEW.n < 3 \
         BEGIN INSERT INTO c(n) VALUES (NEW.n + 1); END",
    );
    exec(&mut db, "INSERT INTO c(n) VALUES (1)");
    // Depth 0: n=1 (top-level plan). Depth 1: n=2 (first action, top-level-populated).
    // Depth 2: n=3 (recompiled action). All three must read m=7 and g=n+100.
    assert_rows(
        &mut db,
        "SELECT n, m, g FROM c ORDER BY n",
        &[
            vec![int(1), int(7), int(101)],
            vec![int(2), int(7), int(102)],
            vec![int(3), int(7), int(103)],
        ],
    );
}

#[test]
fn recursive_trigger_generated_columns_persist_correctly_across_reopen() {
    // The on-disk format: a recompiled-trigger write must produce the SAME on-disk record as a
    // top-level write — STORED present, VIRTUAL omitted. If the depth-2 write had stored the
    // VIRTUAL column, the record would be one slot too wide and a reopen (which re-reads the
    // file, no recompute cached) would mis-map — the corruption a real sqlite3 cross-read
    // would also hit. STORED s and ordinary m must survive; VIRTUAL v must recompute.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "PRAGMA recursive_triggers = ON");
        exec(
            &mut db,
            "CREATE TABLE c(n INTEGER, s INTEGER AS (n * 10) STORED, \
             v INTEGER AS (n + 100) VIRTUAL, m INTEGER DEFAULT 7)",
        );
        exec(
            &mut db,
            "CREATE TRIGGER tr AFTER INSERT ON c WHEN NEW.n < 3 \
             BEGIN INSERT INTO c(n) VALUES (NEW.n + 1); END",
        );
        exec(&mut db, "INSERT INTO c(n) VALUES (1)");
        assert_rows(
            &mut db,
            "SELECT n, s, v, m FROM c ORDER BY n",
            &[
                vec![int(1), int(10), int(101), int(7)],
                vec![int(2), int(20), int(102), int(7)],
                vec![int(3), int(30), int(103), int(7)],
            ],
        );
    }
    {
        let mut db = tmp.open();
        assert_rows(
            &mut db,
            "SELECT n, s, v, m FROM c ORDER BY n",
            &[
                vec![int(1), int(10), int(101), int(7)],
                vec![int(2), int(20), int(102), int(7)],
                vec![int(3), int(30), int(103), int(7)],
            ],
        );
    }
}

// ===========================================================================
// ALTER TABLE DROP COLUMN + generated columns. A VIRTUAL column occupies NO
// physical record slot, so a column's SCHEMA ordinal is not its physical slot
// once a VIRTUAL column precedes/is the target. The DROP COLUMN row rewrite must
// map schema ordinal -> physical slot (skipping virtual columns) and must treat
// dropping a VIRTUAL column as a record NO-OP, or it deletes the WRONG stored
// value — silent data corruption (also an on-disk format corruption).
// ===========================================================================

#[test]
fn drop_virtual_generated_column_preserves_sibling_values() {
    // Physical record is [a, c] (b VIRTUAL, omitted). Dropping b must touch NO stored bytes;
    // using b's schema ordinal (1) as a physical slot would delete c's stored value (20).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b AS (a + 1) VIRTUAL, c INTEGER)");
    exec(&mut db, "INSERT INTO t(a, c) VALUES (10, 20)");
    assert_rows(&mut db, "SELECT a, b, c FROM t", &[vec![int(10), int(11), int(20)]]);
    exec(&mut db, "ALTER TABLE t DROP COLUMN b");
    assert_rows(&mut db, "SELECT a, c FROM t", &[vec![int(10), int(20)]]);
}

#[test]
fn drop_column_after_virtual_generated_removes_correct_value() {
    // Physical record is [a, c, d] (b VIRTUAL, omitted). Dropping c (schema ordinal 2) must
    // remove PHYSICAL slot 1 (c=20), not slot 2 (d=30); a schema-ordinal drop would delete d
    // and leave c's value showing under d.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a INTEGER, b AS (a + 1) VIRTUAL, c INTEGER, d INTEGER)");
    exec(&mut db, "INSERT INTO t2(a, c, d) VALUES (10, 20, 30)");
    assert_rows(
        &mut db,
        "SELECT a, b, c, d FROM t2",
        &[vec![int(10), int(11), int(20), int(30)]],
    );
    exec(&mut db, "ALTER TABLE t2 DROP COLUMN c");
    assert_rows(&mut db, "SELECT a, b, d FROM t2", &[vec![int(10), int(11), int(30)]]);
}

#[test]
fn drop_stored_generated_column_removes_its_slot() {
    // A STORED generated column DOES occupy a physical slot, so dropping it removes that slot
    // and leaves later columns intact. Record is [a, s, c]; dropping s must leave [a, c].
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t3(a INTEGER, s INTEGER AS (a * 2) STORED, c INTEGER)");
    exec(&mut db, "INSERT INTO t3(a, c) VALUES (5, 99)");
    assert_rows(&mut db, "SELECT a, s, c FROM t3", &[vec![int(5), int(10), int(99)]]);
    exec(&mut db, "ALTER TABLE t3 DROP COLUMN s");
    assert_rows(&mut db, "SELECT a, c FROM t3", &[vec![int(5), int(99)]]);
}

#[test]
fn drop_virtual_generated_column_record_valid_across_reopen() {
    // The on-disk format for the DROP: after dropping a VIRTUAL column the on-disk record must
    // be unchanged (b was never stored), so a reopen re-reads a=10, c=20 correctly.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b AS (a + 1) VIRTUAL, c INTEGER)");
        exec(&mut db, "INSERT INTO t(a, c) VALUES (10, 20)");
        exec(&mut db, "ALTER TABLE t DROP COLUMN b");
        assert_rows(&mut db, "SELECT a, c FROM t", &[vec![int(10), int(20)]]);
    }
    {
        let mut db = tmp.open();
        assert_rows(&mut db, "SELECT a, c FROM t", &[vec![int(10), int(20)]]);
    }
}

// ===========================================================================
// Rejected constructs inside a generation expression (gencol.html §2.3: a
// generation expression may not use a subquery, aggregate, or window function).
// The bind refuses these; the error surfaces at CREATE or at first use of the
// table (either is acceptable — the invariant is a LOUD error, never a silent
// wrong value). NON-VACUOUS: `assert_generation_rejected` fails if NEITHER
// statement errors, so a regression to silent acceptance is caught regardless of
// which point the (valid) rejection lands at; each also keeps a positive control
// proving the rejection is construct-specific, not "generated columns broke".
// ===========================================================================

/// Assert a generation expression is rejected LOUDLY: `create` then, if it succeeded,
/// `insert` — one of the two MUST error (per gencol.html the rejection may surface at
/// CREATE or at first use), and the raised error's text must contain `needle`. Panics if
/// BOTH succeed (the vacuous-pass this guards against: a silent-acceptance regression
/// would otherwise slip through). The `Err(e)` bindings infer the engine error type, so
/// no extra import is needed.
fn assert_generation_rejected(create: &str, insert: &str, needle: &str) {
    let mut db = mem();
    let err = match try_exec(&mut db, create) {
        Err(e) => e,
        Ok(()) => match try_exec(&mut db, insert) {
            Err(e) => e,
            Ok(()) => panic!(
                "expected a rejection at CREATE or first use, but BOTH succeeded\n  \
                 create: {create}\n  insert: {insert}"
            ),
        },
    };
    assert!(
        err.to_string().to_lowercase().contains(needle),
        "expected a rejection containing {needle:?}, got: {err}"
    );
}

/// A same-shape but LEGAL generation expression (`b AS (a + 1)`) still works — the positive
/// control proving a rejection above is specific to the forbidden construct, not a blanket
/// "generated columns are broken".
fn assert_plain_generated_column_works() {
    let mut ok = mem();
    exec(&mut ok, "CREATE TABLE ctl(a INTEGER, b AS (a + 1))");
    exec(&mut ok, "INSERT INTO ctl(a) VALUES (1)");
    assert_scalar(&mut ok, "SELECT b FROM ctl", int(2));
}

#[test]
fn reject_subquery_in_generation_expression() {
    assert_generation_rejected(
        "CREATE TABLE t(a INTEGER, b AS ((SELECT 1)))",
        "INSERT INTO t(a) VALUES (1)",
        "subquer",
    );
    assert_plain_generated_column_works();
}

#[test]
fn reject_aggregate_in_generation_expression() {
    assert_generation_rejected(
        "CREATE TABLE t(a INTEGER, b AS (sum(a)))",
        "INSERT INTO t(a) VALUES (1)",
        "aggregate",
    );
    assert_plain_generated_column_works();
}

#[test]
fn reject_window_function_in_generation_expression() {
    // A window function is forbidden too. The generation-expression bind scope carries no
    // windowing context, so `f() OVER (...)` is the same loud "misuse of window function"
    // error as one in WHERE/GROUP BY — never a silent mis-bind. This pins that fail-closed
    // behavior so it cannot regress into an accepted (and wrongly-evaluated) window call.
    assert_generation_rejected(
        "CREATE TABLE t(a INTEGER, b AS (row_number() OVER ()))",
        "INSERT INTO t(a) VALUES (1)",
        "window",
    );
    assert_plain_generated_column_works();
}

// ===========================================================================
// Toposort dependency ordering — deeper coverage of the forward-reference sort.
// ===========================================================================

#[test]
fn generated_columns_resolve_a_multi_level_reverse_chain() {
    // A REVERSE declaration chain d->c->b->a exercises multi-level Kahn ordering: each
    // generated column references the one declared AFTER it, so column order is the exact
    // REVERSE of dependency order. b=a+1, c=b+1, d=c+1 with a=5 => b=6, c=7, d=8.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, d AS (c + 1), c AS (b + 1), b AS (a + 1))");
    exec(&mut db, "INSERT INTO t(a) VALUES (5)");
    assert_rows(
        &mut db,
        "SELECT a, b, c, d FROM t",
        &[vec![int(5), int(6), int(7), int(8)]],
    );
}

#[test]
fn virtual_generated_column_depends_on_stored_and_vice_versa() {
    // A mixed STORED/VIRTUAL dependency in both directions: v (VIRTUAL) reads s (STORED) and
    // w (STORED) reads a base column that another VIRTUAL also uses. On read, the STORED
    // dependency is already materialized in the record and the VIRTUAL is recomputed in
    // dependency order, so v sees s's value. a=4 => s=8 (STORED), v=s+1=9 (VIRTUAL),
    // w=a+100=104 (STORED), x=w+1=105 (VIRTUAL depends on STORED w).
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE t(a INT, s AS (a * 2) STORED, v AS (s + 1) VIRTUAL, \
         w AS (a + 100) STORED, x AS (w + 1) VIRTUAL)",
    );
    exec(&mut db, "INSERT INTO t(a) VALUES (4)");
    assert_rows(
        &mut db,
        "SELECT a, s, v, w, x FROM t",
        &[vec![int(4), int(8), int(9), int(104), int(105)]],
    );
    // And it survives a read-only path with no write-time compute in scope: filter on the
    // VIRTUAL column that depends on a STORED one.
    assert_scalar(&mut db, "SELECT a FROM t WHERE x = 105", int(4));
}

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up on Drop even
// on panic. (Same idiom as the other on-disk conformance files; a `.db`/
// sidecar file is never committed to the repo.)
// ---------------------------------------------------------------------------

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_gencol_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    fn remove_all(&self) {
        let _ = std::fs::remove_file(&self.path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let mut s = self.path.as_os_str().to_os_string();
            s.push(suffix);
            let _ = std::fs::remove_file(PathBuf::from(s));
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        self.remove_all();
    }
}
