//! Behavioral conformance for the `ORDER BY` -> scan-order optimization (the planner's
//! `order_scan`): a single-base-table `SELECT ... ORDER BY <cols> [LIMIT k]` whose order
//! a rowid/index b-tree walk already yields SKIPS the `Sort` and streams the scan, but
//! the returned rows MUST be BYTE-IDENTICAL to the current full stable sort.
//!
//! This file proves the IDENTITY through the real facade (`Connection`) — the plan-shape
//! side (that the `Sort` is actually gone) is pinned by the `order_scan` unit tests in
//! `minisqlite-plan`, which is where `PlanNode` is visible. Two independent oracles per
//! case:
//!
//! 1. **Differential** ([`assert_same_rows`]): the optimized query (`ORDER BY a`) vs a
//!    twin that FORCES a full sort (`ORDER BY a+0` — a computed key the optimizer must
//!    decline) must return identical ordered rows. This compares the optimized path
//!    against the engine's own sort path, so it needs no external reference and it
//!    directly guards the byte-identity contract: an unsound skip (e.g. a reverse index
//!    scan that reverses ties) would diverge from the forced sort here.
//! 2. **Independent reference** (a Rust `slice::sort_by`, STABLE — ties keep insertion
//!    order, matching SQLite's rowid-order tie break): pins the absolute expected order,
//!    including NULL placement (`lang_select.html` §ORDER BY: NULLs sort first for `ASC`,
//!    last for `DESC`) and duplicate-key tie stability.
//!
//! The dataset deliberately carries NULLs and heavy duplicates in the ordered columns so
//! tie stability and NULL placement are exercised, not just a count.

mod conformance;

use std::cmp::Ordering;

use conformance::*;
use minisqlite::{Connection, Value};

// ---- Ground-truth dataset (built here; never read back from the engine) -------

/// One source row. `seq` is the insertion order, kept as a column so a tie-broken
/// result is directly observable: rows equal on the ORDER BY key must come back in
/// ascending `seq` (== ascending rowid, since rows are inserted in `seq` order).
#[derive(Clone)]
struct R {
    seq: i64,
    a: Option<i64>,
    b: i64,
}

/// A deterministic dataset with a SMALL `a` range (heavy duplicates → tie stability is
/// exercised) and ~1-in-13 NULL `a` (NULL placement is exercised). `b` is spread so a
/// composite `(a, b)` order is discriminating. The scrambled `a`/`b` differ from `seq`
/// order, so a wrong scan-order (or a truncate-before-sort) returns the wrong rows.
fn dataset(n: i64) -> Vec<R> {
    (0..n)
        .map(|seq| R {
            seq,
            a: if seq % 13 == 0 { None } else { Some((seq * 5 + 2) % 7) },
            b: (seq * 31 + 7) % 97,
        })
        .collect()
}

/// Create `t(seq, a, b)` and insert `data` in `seq` order, so rowid == seq+1 and the
/// stable-sort tie break (ascending `seq`) is exactly the engine's rowid-order tie break.
fn populate(db: &mut Connection, data: &[R]) {
    exec(db, "CREATE TABLE t(seq INTEGER, a INTEGER, b INTEGER)");
    for r in data {
        let av = r.a.map_or_else(|| "NULL".to_string(), |v| v.to_string());
        exec(db, &format!("INSERT INTO t(seq, a, b) VALUES ({}, {}, {})", r.seq, av, r.b));
    }
}

fn aval(a: &Option<i64>) -> Value {
    a.map_or_else(null, int)
}

// ---- Reference orderings (a STABLE Rust sort over the ground truth) -----------

fn stable<F: FnMut(&R, &R) -> Ordering>(data: &[R], mut cmp: F) -> Vec<R> {
    let mut v = data.to_vec();
    v.sort_by(|x, y| cmp(x, y));
    v
}

/// `ORDER BY a` (ASC default): NULLs first, then values ascending.
fn a_asc(x: &R, y: &R) -> Ordering {
    match (&x.a, &y.a) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(p), Some(q)) => p.cmp(q),
    }
}
/// `ORDER BY a DESC`: values descending, NULLs last (the DESC default placement).
fn a_desc(x: &R, y: &R) -> Ordering {
    match (&x.a, &y.a) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(p), Some(q)) => q.cmp(p),
    }
}

fn proj_a_seq(rows: &[R]) -> Vec<Vec<Value>> {
    rows.iter().map(|r| vec![aval(&r.a), int(r.seq)]).collect()
}

// ---- Differential helper -----------------------------------------------------

/// Run `optimized` and `forced` and assert they return byte-identical ordered rows.
/// `forced` is written to defeat the optimizer (a computed ORDER BY key), so it always
/// takes the full-sort path; `optimized` takes the scan-order path when the rewrite
/// fires. Any divergence (row count, a cell, or the ORDER of tie rows) fails here.
fn assert_same_rows(db: &mut Connection, optimized: &str, forced: &str) {
    let opt = query(db, optimized).rows;
    let force = query(db, forced).rows;
    let same = opt.len() == force.len()
        && opt.iter().zip(&force).all(|(a, b)| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| value_eq(x, y))
        });
    assert!(
        same,
        "differential mismatch (the scan-order skip is NOT byte-identical to the sort)\n  \
         optimized: {optimized}\n  forced:    {forced}\n  optimized rows: {opt:?}\n  \
         forced rows:    {force:?}"
    );
}

// ============================ MUST-SKIP + CORRECT ==============================

#[test]
fn order_by_rowid_asc_and_desc() {
    // rowid is unique => no ties; ASC = forward table walk, DESC = reverse. Rows are
    // inserted in seq order, so ordering by rowid returns seq ascending / descending.
    let data = dataset(60);
    let mut db = mem();
    populate(&mut db, &data);

    let asc: Vec<Vec<Value>> = (0..60).map(|s| vec![int(s)]).collect();
    let desc: Vec<Vec<Value>> = (0..60).rev().map(|s| vec![int(s)]).collect();
    assert_rows(&mut db, "SELECT seq FROM t ORDER BY rowid", &asc);
    assert_rows(&mut db, "SELECT seq FROM t ORDER BY rowid DESC", &desc);
    // Differential vs the forced-sort twins (`rowid+0` is a computed key -> full sort).
    assert_same_rows(&mut db, "SELECT seq FROM t ORDER BY rowid", "SELECT seq FROM t ORDER BY rowid+0");
    assert_same_rows(
        &mut db,
        "SELECT seq FROM t ORDER BY rowid DESC",
        "SELECT seq FROM t ORDER BY rowid+0 DESC",
    );
}

#[test]
fn order_by_integer_primary_key_alias() {
    // `id INTEGER PRIMARY KEY` aliases the rowid, so `ORDER BY id` is the rowid order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(id INTEGER PRIMARY KEY, v TEXT)");
    for (id, v) in [(5, "e"), (2, "b"), (9, "i"), (1, "a"), (7, "g")] {
        exec(&mut db, &format!("INSERT INTO u(id, v) VALUES ({id}, '{v}')"));
    }
    assert_rows(
        &mut db,
        "SELECT id FROM u ORDER BY id",
        &[vec![int(1)], vec![int(2)], vec![int(5)], vec![int(7)], vec![int(9)]],
    );
    assert_rows(
        &mut db,
        "SELECT id FROM u ORDER BY id DESC",
        &[vec![int(9)], vec![int(7)], vec![int(5)], vec![int(2)], vec![int(1)]],
    );
    assert_same_rows(&mut db, "SELECT id, v FROM u ORDER BY id", "SELECT id, v FROM u ORDER BY id+0");
    assert_same_rows(
        &mut db,
        "SELECT id, v FROM u ORDER BY id DESC",
        "SELECT id, v FROM u ORDER BY id+0 DESC",
    );
}

#[test]
fn order_by_indexed_column_asc_with_nulls_and_duplicates() {
    // The headline case: `CREATE INDEX ia ON t(a)` then `SELECT * FROM t ORDER BY a`.
    // NULLs must lead (ASC), duplicate `a` values must keep insertion (rowid) order.
    let data = dataset(60);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");

    assert_rows(&mut db, "SELECT a, seq FROM t ORDER BY a", &proj_a_seq(&stable(&data, a_asc)));
    // Differential vs the forced full sort (byte-identity through the engine).
    assert_same_rows(&mut db, "SELECT a, seq FROM t ORDER BY a", "SELECT a, seq FROM t ORDER BY a+0");
}

#[test]
fn order_by_indexed_column_not_in_select_list_skips_and_is_correct() {
    // Mechanism B with the ORDER BY column HIDDEN from the SELECT list: `ORDER BY a` (a is
    // indexed) while the projection is `seq` / `b`. The rewrite must trace the key through
    // the appended hidden-column projection to the base register, skip the Sort, and still
    // return the PROJECTED column in `a`-order with rowid-ASC ties. (The rowid mechanism's
    // hidden path is covered by `SELECT seq FROM t ORDER BY rowid`; this covers the INDEX
    // mechanism's hidden path, which was otherwise only exercised structurally.)
    let data = dataset(60);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");

    let sorted = stable(&data, a_asc);
    let want_seq: Vec<Vec<Value>> = sorted.iter().map(|r| vec![int(r.seq)]).collect();
    let want_b: Vec<Vec<Value>> = sorted.iter().map(|r| vec![int(r.b)]).collect();
    assert_rows(&mut db, "SELECT seq FROM t ORDER BY a", &want_seq);
    assert_rows(&mut db, "SELECT b FROM t ORDER BY a", &want_b);
    // Differential vs the forced full sort (the hidden `a+0` key is declined -> full sort).
    assert_same_rows(&mut db, "SELECT seq FROM t ORDER BY a", "SELECT seq FROM t ORDER BY a+0");
    assert_same_rows(&mut db, "SELECT b FROM t ORDER BY a", "SELECT b FROM t ORDER BY a+0");
}

#[test]
fn order_by_binary_text_index_skips_and_is_correct() {
    // The only end-to-end case that SKIPS via a BINARY index over TEXT: a default-collation
    // TEXT column with an index on it. `ORDER BY s` is servable because the index's byte
    // order equals the Sort's Binary text order. The `+0` twin can't force-sort text (it
    // coerces text to 0), so the forced twin is `s||''` — a computed, NULL-PRESERVING
    // identity key the optimizer declines (`NULL||''` is NULL, `'x'||''` is `'x'`). Mixed
    // case + duplicates + a NULL exercise byte order (uppercase < lowercase in BINARY), tie
    // stability, and NULL-first placement.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ct(seq INTEGER, s TEXT)");
    let rows: [(i64, Option<&str>); 8] = [
        (0, Some("banana")),
        (1, Some("Apple")),
        (2, Some("cherry")),
        (3, Some("APPLE")),
        (4, None),
        (5, Some("banana")),
        (6, Some("Banana")),
        (7, Some("apple")),
    ];
    for &(seq, s) in &rows {
        match s {
            Some(v) => exec(&mut db, &format!("INSERT INTO ct(seq, s) VALUES ({seq}, '{v}')")),
            None => exec(&mut db, &format!("INSERT INTO ct(seq, s) VALUES ({seq}, NULL)")),
        }
    }
    exec(&mut db, "CREATE INDEX ics ON ct(s)");

    // Independent reference: BINARY byte order, NULLs first, ties by seq (rowid) ascending —
    // exactly the stable sort the skipped scan must reproduce.
    let mut sorted = rows.to_vec();
    sorted.sort_by(|x, y| match (&x.1, &y.1) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(p), Some(q)) => p.as_bytes().cmp(q.as_bytes()),
    });
    let want_s: Vec<Vec<Value>> =
        sorted.iter().map(|&(_, s)| vec![s.map_or_else(null, text)]).collect();
    let want_seq: Vec<Vec<Value>> = sorted.iter().map(|&(seq, _)| vec![int(seq)]).collect();

    // `s` projected, and `s` hidden (project `seq`) — both must skip via the index.
    assert_rows(&mut db, "SELECT s FROM ct ORDER BY s", &want_s);
    assert_rows(&mut db, "SELECT seq FROM ct ORDER BY s", &want_seq);
    // Differential vs the forced full sort (`s||''` keeps the Sort, NULL-preserving).
    assert_same_rows(&mut db, "SELECT s FROM ct ORDER BY s", "SELECT s FROM ct ORDER BY s||''");
    assert_same_rows(&mut db, "SELECT seq FROM ct ORDER BY s", "SELECT seq FROM ct ORDER BY s||''");
}

#[test]
fn order_by_indexed_column_desc_is_correct_even_though_sort_is_kept() {
    // DESC over a (dup-valued) secondary index KEEPS the Sort (a reverse index walk would
    // reverse ties vs the stable sort). The result must still be correct AND identical to
    // the forced-sort twin — this locks in that DESC is NOT (unsoundly) turned into a
    // reverse scan, which would order duplicate-`a` ties by rowid DESC instead of ASC.
    let data = dataset(60);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");

    assert_rows(&mut db, "SELECT a, seq FROM t ORDER BY a DESC", &proj_a_seq(&stable(&data, a_desc)));
    assert_same_rows(
        &mut db,
        "SELECT a, seq FROM t ORDER BY a DESC",
        "SELECT a, seq FROM t ORDER BY a+0 DESC",
    );
}

#[test]
fn range_on_indexed_column_then_order_same_column() {
    // `WHERE a >= 3 ORDER BY a`: the index range scan already emits `a` ascending.
    let data = dataset(60);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");

    let want: Vec<R> = stable(&data, a_asc).into_iter().filter(|r| r.a.is_some_and(|v| v >= 3)).collect();
    assert_rows(&mut db, "SELECT a, seq FROM t WHERE a >= 3 ORDER BY a", &proj_a_seq(&want));
    assert_same_rows(
        &mut db,
        "SELECT a, seq FROM t WHERE a >= 3 ORDER BY a",
        "SELECT a, seq FROM t WHERE a >= 3 ORDER BY a+0",
    );
}

#[test]
fn eq_prefix_then_order_on_next_column() {
    // Composite `(a, b)` index, ONLY index on the table so `WHERE a = 1` must seek it:
    // `ORDER BY b` continues the index after the `a=` equality prefix. ASC skips the Sort;
    // the result (and its `b` tie order) must match the forced sort.
    let data = dataset(90);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX iab ON t(a, b)");

    // Reference: the a=1 rows, ordered by b ascending, ties by seq (rowid) ascending.
    let mut want: Vec<R> = data.iter().filter(|r| r.a == Some(1)).cloned().collect();
    want.sort_by(|x, y| x.b.cmp(&y.b).then(x.seq.cmp(&y.seq)));
    let want_rows: Vec<Vec<Value>> = want.iter().map(|r| vec![int(r.b), int(r.seq)]).collect();
    assert_rows(&mut db, "SELECT b, seq FROM t WHERE a = 1 ORDER BY b", &want_rows);
    assert_same_rows(
        &mut db,
        "SELECT b, seq FROM t WHERE a = 1 ORDER BY b",
        "SELECT b, seq FROM t WHERE a = 1 ORDER BY b+0",
    );
    // The composite index also serves `ORDER BY a, b` over the whole table (== full index).
    assert_same_rows(
        &mut db,
        "SELECT a, b, seq FROM t ORDER BY a, b",
        "SELECT a, b, seq FROM t ORDER BY a+0, b+0",
    );
}

#[test]
fn order_by_indexed_column_with_limit_on_a_large_table() {
    // 500 rows, indexed `a`, `ORDER BY a LIMIT 3` and `LIMIT 3 OFFSET 2`: the scan streams
    // and the Limit early-stops — the first rows must match the full-sort reference exactly.
    let data = dataset(500);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");

    let sorted = stable(&data, a_asc);
    assert_rows(&mut db, "SELECT a, seq FROM t ORDER BY a LIMIT 3", &proj_a_seq(&sorted[..3]));
    assert_rows(
        &mut db,
        "SELECT a, seq FROM t ORDER BY a LIMIT 3 OFFSET 2",
        &proj_a_seq(&sorted[2..5]),
    );
    // Differential across a range of k, including past the row count.
    for lim in ["0", "1", "3", "50", "499", "500", "1000"] {
        assert_same_rows(
            &mut db,
            &format!("SELECT a, seq FROM t ORDER BY a LIMIT {lim}"),
            &format!("SELECT a, seq FROM t ORDER BY a+0 LIMIT {lim}"),
        );
    }
    assert_same_rows(
        &mut db,
        "SELECT a, seq FROM t ORDER BY a LIMIT 7 OFFSET 4",
        "SELECT a, seq FROM t ORDER BY a+0 LIMIT 7 OFFSET 4",
    );
}

// ============================ MUST-NOT-SKIP (Sort kept, result still correct) ===

#[test]
fn order_by_nocase_column_keeps_sort_and_is_correct() {
    // A NOCASE-declared column (and an explicit COLLATE NOCASE) cannot be served by the
    // BINARY index; the Sort stays and folds case. `c` has case-variant duplicates so
    // BINARY (uppercase-first) and NOCASE differ.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE c(seq INTEGER, s TEXT COLLATE NOCASE)");
    for (seq, s) in ["banana", "Apple", "cherry", "APPLE", "Banana"].iter().enumerate() {
        exec(&mut db, &format!("INSERT INTO c(seq, s) VALUES ({seq}, '{s}')"));
    }
    exec(&mut db, "CREATE INDEX ics ON c(s)");
    // NOCASE order: apple(1), APPLE(3), banana(0), Banana(4), cherry(2) — case-folded,
    // ties (apple/APPLE, banana/Banana) by insertion order.
    assert_rows(
        &mut db,
        "SELECT seq FROM c ORDER BY s",
        &[vec![int(1)], vec![int(3)], vec![int(0)], vec![int(4)], vec![int(2)]],
    );
    // Explicit COLLATE NOCASE on a fresh BINARY column matches the same fold.
    assert_rows(
        &mut db,
        "SELECT seq FROM c ORDER BY s COLLATE NOCASE",
        &[vec![int(1)], vec![int(3)], vec![int(0)], vec![int(4)], vec![int(2)]],
    );
}

#[test]
fn order_by_non_leading_index_column_keeps_sort_and_is_correct() {
    // Only index is `(a, b)`; `ORDER BY b` (no `a=` equality) cannot use it. Result must
    // still be correctly ordered by `b`.
    let data = dataset(40);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX iab ON t(a, b)");
    assert_same_rows(&mut db, "SELECT b, seq FROM t ORDER BY b", "SELECT b, seq FROM t ORDER BY b+0");
}

#[test]
fn order_by_computed_expression_keeps_sort_and_is_correct() {
    let data = dataset(40);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");
    // `a+0` is a computed key (the optimizer declines it — proven in the plan-shape unit
    // tests), so the Sort is kept. `a+0` equals `a` on our integer/NULL data, so the
    // result must equal the independent ASC reference (NULLs first, ties by rowid).
    assert_rows(&mut db, "SELECT a, seq FROM t ORDER BY a+0", &proj_a_seq(&stable(&data, a_asc)));
}

#[test]
fn mixed_direction_keeps_sort_and_is_correct() {
    // `ORDER BY a ASC, b DESC` over an all-ascending `(a, b)` index: mixed directions, one
    // walk cannot serve both, so the Sort stays. Verify against an independent reference.
    let data = dataset(50);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX iab ON t(a, b)");
    let mut want = data.clone();
    // a ascending (NULLs first), ties by b DESCending, then by seq (rowid) ascending.
    want.sort_by(|x, y| a_asc(x, y).then(y.b.cmp(&x.b)).then(x.seq.cmp(&y.seq)));
    let want_rows: Vec<Vec<Value>> = want.iter().map(|r| vec![aval(&r.a), int(r.b), int(r.seq)]).collect();
    assert_rows(&mut db, "SELECT a, b, seq FROM t ORDER BY a ASC, b DESC", &want_rows);
}

#[test]
fn nulls_last_on_asc_index_keeps_sort_and_is_correct() {
    // An ASC index scan is naturally NULLS FIRST; an explicit `NULLS LAST` contradicts it,
    // so the Sort stays and pushes NULLs to the end.
    let data = dataset(40);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");
    let mut want = data.clone();
    want.sort_by(|x, y| match (&x.a, &y.a) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(p), Some(q)) => p.cmp(q),
    });
    assert_rows(&mut db, "SELECT a, seq FROM t ORDER BY a NULLS LAST", &proj_a_seq(&want));
}

#[test]
fn aggregate_and_distinct_keep_sort_and_are_correct() {
    let data = dataset(40);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia ON t(a)");
    // Aggregate: GROUP BY a ORDER BY a — the aggregate re-orders rows, so the Sort stays.
    // Distinct group keys: NULL first, then 0..6 (present values), each once.
    assert_rows(
        &mut db,
        "SELECT a FROM t GROUP BY a ORDER BY a",
        &[vec![null()], vec![int(0)], vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)], vec![int(5)], vec![int(6)]],
    );
    // DISTINCT over `a`: same distinct set, ordered.
    assert_rows(
        &mut db,
        "SELECT DISTINCT a FROM t ORDER BY a",
        &[vec![null()], vec![int(0)], vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)], vec![int(5)], vec![int(6)]],
    );
}

#[test]
fn join_order_keeps_sort_and_is_correct() {
    // An ORDER BY over a two-table join is not a single base table, so the Sort stays.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(id INTEGER, x INTEGER)");
    exec(&mut db, "CREATE TABLE r(id INTEGER, y INTEGER)");
    for (id, x) in [(1, 30), (2, 10), (3, 20)] {
        exec(&mut db, &format!("INSERT INTO l VALUES ({id}, {x})"));
    }
    for (id, y) in [(1, 100), (2, 200), (3, 300)] {
        exec(&mut db, &format!("INSERT INTO r VALUES ({id}, {y})"));
    }
    exec(&mut db, "CREATE INDEX lx ON l(x)");
    assert_rows(
        &mut db,
        "SELECT l.x, r.y FROM l JOIN r ON l.id = r.id ORDER BY l.x",
        &[vec![int(10), int(200)], vec![int(20), int(300)], vec![int(30), int(100)]],
    );
}

#[test]
fn partial_index_is_not_used_for_order_by() {
    // A PARTIAL index covers only some rows, so a full index scan would MISS rows. The
    // planner must NOT use it to serve ORDER BY; the result must include EVERY row in the
    // right order (a wrong skip here would silently drop the `a <= 0` / NULL rows).
    let data = dataset(50);
    let mut db = mem();
    populate(&mut db, &data);
    exec(&mut db, "CREATE INDEX ia_pos ON t(a) WHERE a > 0");
    assert_rows(&mut db, "SELECT a, seq FROM t ORDER BY a", &proj_a_seq(&stable(&data, a_asc)));
    // And it agrees with the forced full sort (which never considers the index at all).
    assert_same_rows(&mut db, "SELECT a, seq FROM t ORDER BY a", "SELECT a, seq FROM t ORDER BY a+0");
}
