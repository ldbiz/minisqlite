//! Conformance battery for CORRELATED-subquery MEMOIZATION, end to end through the
//! `minisqlite` facade.
//!
//! The engine memoizes a correlated scalar / `EXISTS` / `IN` subquery by the VALUES of the
//! outer columns it references, so a low-cardinality correlation runs the subplan once per
//! DISTINCT key instead of once per outer row (the O(n^2) -> O(n) win). Correctness is
//! paramount: a memo may only change COST, never the answer. Every expected value here is
//! the SQL-correct one, transcribed from the SQLite documentation — not from what this
//! engine happens to return:
//!
//!   * `spec/sqlite-doc/lang_expr.html` §11 "Subquery Expressions" — a scalar subquery is
//!     the first row of its SELECT, NULL if it returns no rows.
//!   * `spec/sqlite-doc/lang_expr.html` §10 "The EXISTS operator" — 1 if the SELECT would
//!     return a row, else 0; NULL rows are not special.
//!   * `spec/sqlite-doc/lang_expr.html` §8 "The IN operator" — membership with the
//!     documented three-valued NULL matrix.
//!   * `spec/sqlite-doc/lang_expr.html` §12 "Correlated Subqueries" — a correlated subquery
//!     "is reevaluated each time its result is required"; the memo is a pure optimization
//!     that must preserve that observable result exactly.
//!   * `spec/sqlite-doc/lang_createtable.html` (affinity) — a column declared with NO type
//!     has NONE (BLOB) affinity, so an inserted Integer / Real keeps its storage class; the
//!     correlation key is storage-class exact, so `2` (int) and `2.0` (real) are DISTINCT.
//!
//! The two GUARDS that keep the memo from ever serving a stale answer are exercised by
//! their bite tests below (each would produce a wrong result if its guard were removed):
//!
//!   * (a) a MUTATING statement is never memoized (`dml_correlated_subquery_*`): the
//!     enclosing write changes the table the subquery reads, so a per-key memo would be
//!     stale — the engine re-runs the subquery per row (its live, row-by-row semantics).
//!   * (b) a VOLATILE correlated subquery (containing `random()`) is never memoized
//!     (`volatile_correlated_subquery_*`): it must re-draw per outer row, not repeat one
//!     draw across a key.
//!
//! A case that reveals an engine bug is left as a genuine failing assertion rather than
//! weakened to pass. `Value` has no `PartialEq`, so every check goes through the shared
//! harness (`assert_rows` / `assert_scalar` / `value_eq`); never compare a `Value` with `==`.

mod conformance;

use conformance::*;
use minisqlite::Connection;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Fixture. `o` is the OUTER table whose correlation column `g` REPEATS (so the
// second+ row with a given key actually HITS the memo) and includes a NULL key
// (two NULL-g rows share one key). `dim` is the inner table a correlated
// subquery reads, keyed by `g`.
//
//   o(id, g, h):  (1,1,100) (2,1,100) (3,2,200) (4,NULL,300) (5,NULL,300) (6,2,200)
//   dim(g, val):  (1,10) (1,11) (2,20) (NULL,99)
// ---------------------------------------------------------------------------

fn fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, g INTEGER, h INTEGER)");
    exec(
        &mut db,
        "INSERT INTO o VALUES (1,1,100),(2,1,100),(3,2,200),(4,NULL,300),(5,NULL,300),(6,2,200)",
    );
    exec(&mut db, "CREATE TABLE dim(g INTEGER, val INTEGER)");
    exec(&mut db, "INSERT INTO dim VALUES (1,10),(1,11),(2,20),(NULL,99)");
    db
}

// ---------------------------------------------------------------------------
// (1) CORRECTNESS BATTERY — the memoized answer is the SQL-correct answer, with
// multiple outer rows SHARING a key (so the memo genuinely hits), NULL keys,
// multi-column keys, storage-class-distinct keys, and all three shapes.
// ---------------------------------------------------------------------------

#[test]
fn correlated_scalar_count_shared_key_and_null_key() {
    // Scalar shape. `(SELECT count(*) FROM dim d WHERE d.g = o.g)` per outer row:
    //   g=1 -> 2 matching dim rows (rows 1,2 share this key), g=2 -> 1 (rows 3,6),
    //   g=NULL -> 0 because `d.g = NULL` is never true (rows 4,5 share the NULL key).
    // The memo collapses the three keys; the answer must be identical to a per-row re-run.
    let mut db = fixture();
    assert_rows(
        &mut db,
        "SELECT id, (SELECT count(*) FROM dim d WHERE d.g = o.g) FROM o ORDER BY id",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(2)],
            vec![int(3), int(1)],
            vec![int(4), int(0)],
            vec![int(5), int(0)],
            vec![int(6), int(1)],
        ],
    );
}

#[test]
fn correlated_scalar_reads_outer_column_per_row() {
    // A correlated scalar that READS the outer key `(SELECT o.g * 100)` (no FROM). The memo
    // is keyed on `g`, and the result for a key is that key's transform — same for every row
    // sharing the key, NULL for the NULL-key rows (NULL*100 = NULL). Proves the memoized
    // value tracks the outer key, not a frozen first row.
    let mut db = fixture();
    assert_rows(
        &mut db,
        "SELECT id, (SELECT o.g * 100) FROM o ORDER BY id",
        &[
            vec![int(1), int(100)],
            vec![int(2), int(100)],
            vec![int(3), int(200)],
            vec![int(4), null()],
            vec![int(5), null()],
            vec![int(6), int(200)],
        ],
    );
}

#[test]
fn correlated_exists_shared_key_and_null_key() {
    // EXISTS shape. A row is kept iff dim has a matching g: g=1 and g=2 match, g=NULL never
    // matches (`d.g = NULL` is never true). Rows 1,2 share key 1; rows 3,6 share key 2; the
    // memo serves the second of each pair.
    let mut db = fixture();
    assert_rows(
        &mut db,
        "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM dim d WHERE d.g = o.g) ORDER BY id",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(6)]],
    );
}

#[test]
fn correlated_in_shares_set_but_probes_fresh() {
    // IN shape, the "shared candidate set, fresh probe" property: same-key rows reuse ONE
    // materialized candidate set while each probes with its own subject. Over category 1
    // (rows sharing the key) the set is {10, 20}; the varying prices 10, 99, 20 give
    // true, false, true — a memo that also froze the PROBE would corrupt this.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE items(cat INTEGER, price INTEGER)");
    exec(&mut db, "INSERT INTO items VALUES (1,10),(1,20),(2,30)");
    exec(&mut db, "CREATE TABLE ord(id INTEGER, cat INTEGER, p INTEGER)");
    exec(&mut db, "INSERT INTO ord VALUES (1,1,10),(2,1,99),(3,1,20),(4,2,30)");
    assert_rows(
        &mut db,
        "SELECT id FROM ord o WHERE o.p IN (SELECT price FROM items i WHERE i.cat = o.cat) ORDER BY id",
        &[vec![int(1)], vec![int(3)], vec![int(4)]],
    );
}

#[test]
fn correlated_multicolumn_key() {
    // A MULTI-COLUMN correlation key (g AND h). A self-count of rows sharing (g, h):
    //   (g,h)=(1,1) -> 2 (rows 1,2), (1,2) -> 1 (row 3). The memo keys on BOTH columns, so
    // rows 1,2 collapse but row 3 (differing in h) does not. Under-keying (dropping h) would
    // wrongly give 3 for every row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mc(id INTEGER, g INTEGER, h INTEGER)");
    exec(&mut db, "INSERT INTO mc VALUES (1,1,1),(2,1,1),(3,1,2)");
    assert_rows(
        &mut db,
        "SELECT id, (SELECT count(*) FROM mc x WHERE x.g = mc.g AND x.h = mc.h) FROM mc ORDER BY id",
        &[vec![int(1), int(2)], vec![int(2), int(2)], vec![int(3), int(1)]],
    );
}

#[test]
fn correlated_key_is_storage_class_exact() {
    // The correlation key is STORAGE-CLASS EXACT: an outer key Integer(2) and Real(2.0) are
    // DIFFERENT keys, so a correlated `(SELECT typeof(sc.k))` returns 'integer' vs 'real'
    // rather than collapsing to one cached class. A value-folding key (2 == 2.0) would serve
    // the second row the first's 'integer' — the regression this pins.
    let mut db = mem();
    // `k` has NO declared type => NONE affinity, so 2 stays Integer and 2.0 stays Real.
    exec(&mut db, "CREATE TABLE sc(id INTEGER, k)");
    exec(&mut db, "INSERT INTO sc VALUES (1, 2), (2, 2.0)");
    // Precondition: the two rows really are distinct storage classes (guards against an
    // affinity that would coerce them and silently defeat the test).
    assert_rows(
        &mut db,
        "SELECT typeof(k) FROM sc ORDER BY id",
        &[vec![text("integer")], vec![text("real")]],
    );
    // The correlated memo must keep them distinct.
    assert_rows(
        &mut db,
        "SELECT id, (SELECT typeof(sc.k)) FROM sc ORDER BY id",
        &[vec![int(1), text("integer")], vec![int(2), text("real")]],
    );
}

#[test]
fn correlated_nested_subquery_key_threads_grandparent_column() {
    // NESTED (doubly) correlation — the invariant the whole feature rests on. The OUTER
    // correlated subquery references the grandparent column `o.h` ONLY through an inner
    // subquery `(SELECT o.h)`, never directly. For the memo to be correct, the planner's
    // correlation analysis must thread that grandparent register into the OUTER subquery's
    // key TRANSITIVELY; otherwise the outer memo, keyed only on the directly-referenced
    // `o.g`, would collapse two rows that differ in `h` and serve a STALE cached answer.
    // Activating the memo is exactly what turns such an analysis gap from merely SLOW into
    // WRONG (a per-row re-run reads each row's own `o.h` and is always correct), so this pins
    // the single load-bearing dependency of the optimization.
    //
    // `o.g` is constant (1) so ONLY `h` distinguishes the rows, and row 3 repeats row 1's
    // (g,h) = (1,10) so the outer memo genuinely HITS there. count of `d` where d.g=o.g AND
    // d.v=o.h:  row1 h=10 -> 2,  row2 h=20 -> 1,  row3 h=10 -> 2  =>  [2,1,2].
    // A memo keyed only on `g` (grandparent `h` dropped) would serve row 2 the cached 2,
    // giving a wrong [2,2,2].
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, g INTEGER, h INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (1,1,10),(2,1,20),(3,1,10)");
    exec(&mut db, "CREATE TABLE d(g INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO d VALUES (1,10),(1,10),(1,20)");
    assert_rows(
        &mut db,
        "SELECT id, (SELECT count(*) FROM d WHERE d.g = o.g AND d.v = (SELECT o.h)) \
         FROM o ORDER BY id",
        &[vec![int(1), int(2)], vec![int(2), int(1)], vec![int(3), int(2)]],
    );
}

// ---------------------------------------------------------------------------
// (2) DML EDGE — GUARD (a): a mutating statement is never memoized. The
// subquery reads a column the UPDATE writes, so a per-key memo would serve a
// STALE count; the engine re-runs per row (live, row-by-row) and gets the
// correct multiset. This test goes WRONG if guard (a) is removed.
// ---------------------------------------------------------------------------

#[test]
fn dml_correlated_subquery_not_memoized_when_mutating() {
    // Two rows share key g=5. `UPDATE t SET tag = (SELECT count(*) FROM t x WHERE x.g=t.g
    // AND x.tag=0)` reads `tag`, which the UPDATE itself writes. Processing row by row: the
    // first row sees both tags=0 (count 2) and becomes 2; the second row then sees only
    // itself still 0 (count 1) and becomes 1. So the tags are the multiset {1, 2}. A memo
    // (guard a missing) would freeze the g=5 count at 2 and give {2, 2} — the stale-DML bug.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, g INTEGER, tag INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1,5,0),(2,5,0)");
    exec(&mut db, "UPDATE t SET tag = (SELECT count(*) FROM t x WHERE x.g = t.g AND x.tag = 0)");
    // ORDER BY tag makes the multiset {1,2} an ordered [1],[2]; a frozen memo would give
    // [2],[2].
    assert_rows(&mut db, "SELECT tag FROM t ORDER BY tag", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn dml_correlated_subquery_stable_column_is_unaffected() {
    // The companion to the bite test: when the correlated subquery reads a column the UPDATE
    // does NOT touch, the per-row answer is stable and both rows get the same value. Here the
    // subquery counts by `g` (unchanged) while `tag` is written, so both g=5 rows get 2.
    // Whether or not the memo were active, this answer is the same — it documents that the
    // guard costs nothing on the common, safe case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, g INTEGER, tag INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1,5,0),(2,5,0),(3,9,0)");
    exec(&mut db, "UPDATE t SET tag = (SELECT count(*) FROM t x WHERE x.g = t.g)");
    assert_rows(
        &mut db,
        "SELECT id, tag FROM t ORDER BY id",
        &[vec![int(1), int(2)], vec![int(2), int(2)], vec![int(3), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// (3) VOLATILE — GUARD (b): a correlated subquery containing random() is never
// memoized; it re-draws per outer row. This test goes WRONG if guard (b) is
// removed (the memo would repeat one draw per correlation key).
// ---------------------------------------------------------------------------

#[test]
fn volatile_correlated_subquery_reruns_per_outer_row() {
    // g = id % 3 gives just 3 distinct correlation keys over 30 rows. If a correlated
    // `(SELECT random() ... WHERE x.g = o.g LIMIT 1)` were wrongly memoized by g, it would
    // produce only 3 distinct values (one per key). Because random() is non-deterministic,
    // guard (b) forces a re-run per outer row, so all 30 draws are distinct (a good PRNG
    // does not repeat within 30 draws). count(DISTINCT ...) == 30 proves it was NOT memoized.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, g INTEGER)");
    for id in 0..30 {
        exec(&mut db, &format!("INSERT INTO t VALUES ({id}, {})", id % 3));
    }
    assert_scalar(
        &mut db,
        "SELECT count(DISTINCT (SELECT random() FROM t x WHERE x.g = o.g LIMIT 1)) FROM t o",
        int(30),
    );
}

// ---------------------------------------------------------------------------
// (4) PERFORMANCE — the O(n^2) -> O(n) win, end to end through the real planner
// and executor. The bench workload `SELECT count(*) FROM t a WHERE
// a.k=(SELECT max(k) FROM t b WHERE b.g=a.g)` correlates on a low-cardinality
// column (g = id % 10), so the memo collapses ~n subplan runs to ~10. Without
// the memo this is n outer rows x an n-row inner scan = O(n^2) and HANGS at
// scale. We prove it no longer hangs (a size that O(n^2) would blow far past a
// generous timeout) and record the per-size timings so the ~linear scaling is
// on the record.
// ---------------------------------------------------------------------------

const PERF_QUERY: &str =
    "SELECT count(*) FROM t a WHERE a.k = (SELECT max(k) FROM t b WHERE b.g = a.g)";

/// Build an `n`-row table `t(id, k, g)` with a LOW-cardinality `g` (id % 10) and NO index on
/// `g` — so the inner `WHERE b.g = a.g` is a full scan and the memo's per-key collapse is the
/// ONLY thing that makes the correlated query sub-quadratic. Inserted in batches so data-gen
/// stays linear.
fn build_perf_table(db: &mut Connection, n: usize) {
    exec(db, "CREATE TABLE t(id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)");
    let mut i = 0usize;
    while i < n {
        let batch = (n - i).min(1_000);
        let mut sql = String::from("INSERT INTO t(id, k, g) VALUES ");
        for j in 0..batch {
            let id = i + j;
            if j > 0 {
                sql.push(',');
            }
            sql.push_str(&format!("({id},{id},{})", id % 10));
        }
        exec(db, &sql);
        i += batch;
    }
}

/// Run [`PERF_QUERY`] over a fresh `n`-row table on a WORKER THREAD, returning its wall-clock
/// duration, or `None` if it did not finish within `timeout`. A `None` is the O(n^2)
/// regression: the memo stopped engaging, so the query fell back to a per-row inner scan. The
/// `Connection` is built inside the thread, so nothing crosses the boundary but the size in
/// and the timing out — and the timeout keeps a regression from HANGING the whole suite
/// (a result that never arrives is a wrong result).
fn time_perf_query(n: usize, timeout: Duration) -> Option<Duration> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut db = mem();
        build_perf_table(&mut db, n);
        let start = Instant::now();
        let qr = query(&mut db, PERF_QUERY);
        let elapsed = start.elapsed();
        std::hint::black_box(qr.rows.len());
        let _ = tx.send(elapsed);
    });
    rx.recv_timeout(timeout).ok()
}

#[test]
fn correlated_subquery_completes_and_scales_subquadratically() {
    // The HARD assertion is completion within a generous timeout at the top size: at
    // n=25_000 an O(n^2) plan is ~6.25e8 inner-row reads and cannot finish in 20s, while the
    // memoized O(n) plan (~10 keys) finishes in well under a second. The per-size timings are
    // printed (run `cargo test -- --nocapture`) so the ~linear growth is visible; a strict
    // ratio assertion is deliberately avoided as timing-flaky on a shared host — the timeout
    // is the robust regression catch.
    let timeout = Duration::from_secs(20);
    let sizes = [6_250usize, 12_500, 25_000];
    let mut timings = Vec::new();
    for &n in &sizes {
        let elapsed = time_perf_query(n, timeout).unwrap_or_else(|| {
            panic!(
                "correlated_subquery did NOT finish within {timeout:?} at n={n}: this is the \
                 O(n^2) regression the memo exists to prevent (a low-cardinality correlation \
                 must stay ~linear). Check that context::correlated_memo_eligible still \
                 engages for a read-only, deterministic, correlated subquery."
            )
        });
        eprintln!("correlated_subquery n={n:>6}: {:>8.1} ms", elapsed.as_secs_f64() * 1000.0);
        timings.push(elapsed.as_secs_f64());
    }
    // Informational scaling signal (not a hard gate): n quadrupled (6_250 -> 25_000), so a
    // linear plan grows ~4x and a quadratic one ~16x. A generous ceiling flags a gross
    // quadratic regression that still (barely) beat the timeout, with a floor to damp
    // sub-millisecond noise at the smallest size.
    let base = timings[0].max(0.010);
    let ratio = timings[2] / base;
    eprintln!("correlated_subquery scaling (25_000 / 6_250): {ratio:.1}x (linear ~4x)");
    assert!(
        ratio <= 12.0,
        "correlated_subquery scaling looks quadratic: n x4 but time x{ratio:.1} \
         ({:.1} ms -> {:.1} ms); the memo should keep it ~linear",
        timings[0] * 1000.0,
        timings[2] * 1000.0,
    );
}
