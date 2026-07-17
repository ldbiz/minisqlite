//! Local performance + durability harness covering scalability and durability.
//! Run it with `cargo bench`.
//!
//! It builds each representative workload at growing sizes and reports wall-clock,
//! process peak RSS, and a heap-allocation probe, then exercises the durability
//! round-trip (write, reopen, recover), so a quadratic plan, a clone-the-world
//! statement, or a materialized intermediate shows up here. This is measurement,
//! not pass/fail: each workload runs under `catch_unwind`, so until the engine can
//! run one its cell reads `unimplemented` rather than aborting the run.
//!
//! std-only on purpose (the workspace has no external dependencies): timing is
//! `Instant`, peak RSS reads `/proc/self/status` (`VmHWM`), and heap bytes come
//! from a counting global allocator.

use minisqlite::{Connection, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Sizes the scalability sweep. The top size is where an O(n^2) plan or a
/// per-statement whole-database copy separates from a streaming, index-aware one.
const SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

// --- heap-allocation probe (counts live and peak bytes) ----------------------

struct CountingAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset_peak_heap() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak_heap_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Process peak resident set, in KiB, from `/proc/self/status` (`VmHWM`). It is
/// process-monotonic, so it reads as "peak so far"; isolate per-workload peaks by
/// running each in controlled conditions.
fn peak_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0);
        }
    }
    0
}

// --- workload construction + queries -----------------------------------------

/// Build a table of `n` rows: `t(id INTEGER PRIMARY KEY, k INTEGER, v TEXT, g INTEGER)`
/// with a secondary index on `k`, inserted in batches so data-gen itself stays linear.
fn build(db: &mut Connection, n: usize) {
    db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, k INTEGER, v TEXT, g INTEGER)").unwrap();
    db.execute("CREATE INDEX t_k ON t(k)").unwrap();
    let mut i = 0usize;
    while i < n {
        let batch = (n - i).min(1_000);
        let mut sql = String::from("INSERT INTO t(id, k, v, g) VALUES ");
        for j in 0..batch {
            let id = i + j;
            if j > 0 {
                sql.push(',');
            }
            sql.push_str(&format!("({id},{},'r{id}',{})", id % 977, id % 10));
        }
        db.execute(&sql).unwrap();
        i += batch;
    }
}

/// The five access shapes the scalability sweep exercises.
const WORKLOADS: &[(&str, &str)] = &[
    ("point_lookup_indexed", "SELECT v FROM t WHERE k = 500"),
    ("range_scan", "SELECT count(*) FROM t WHERE k BETWEEN 100 AND 200"),
    ("equi_join", "SELECT count(*) FROM t a JOIN t b ON a.k = b.k"),
    ("group_by", "SELECT g, count(*) FROM t GROUP BY g"),
    ("correlated_subquery", "SELECT count(*) FROM t a WHERE a.k = (SELECT max(k) FROM t b WHERE b.g = a.g)"),
];

/// Measure one query over a freshly built `n`-row table. Returns
/// `(elapsed_ms, peak_heap_bytes)` or `None` if the engine is not implemented yet.
fn measure(query: &str, n: usize) -> Option<(f64, usize)> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut db = Connection::open_in_memory().expect("open_in_memory");
        build(&mut db, n);
        reset_peak_heap();
        let start = Instant::now();
        let result = db.query(query).expect("query");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        // Touch the result so a lazy engine cannot skip the work.
        std::hint::black_box(result.rows.len());
        (elapsed, peak_heap_bytes())
    }))
    .ok()
}

/// Durability round-trip: committed data survives a reopen; a rolled-back mutation
/// leaves no trace. This local check is the clean-recovery half (it does not inject
/// a crash mid-transaction).
fn durability_roundtrip() -> Result<bool, ()> {
    catch_unwind(AssertUnwindSafe(|| {
        let path = std::env::temp_dir().join(format!("minisqlite-bench-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);

        {
            let mut db = Connection::open(&path).expect("open");
            db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)").unwrap();
            db.execute("INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
            db.execute("BEGIN").unwrap();
            db.execute("DELETE FROM t").unwrap();
            db.execute("ROLLBACK").unwrap();
        }

        let mut db = Connection::open(&path).expect("reopen");
        let rows = db.query("SELECT count(*) FROM t").expect("query").rows;
        let _ = std::fs::remove_file(&path);
        matches!(rows.first().and_then(|r| r.first()), Some(Value::Integer(3)))
    }))
    .map_err(|_| ())
}

fn main() {
    // Suppress the panic backtrace spam from the expected pre-implementation
    // `unimplemented!()` panics; we report them as `unimplemented` cells instead.
    std::panic::set_hook(Box::new(|_| {}));

    println!("scalability (wall-clock ms / peak heap KiB), sizes {SIZES:?}");
    println!("{:<24}{}", "workload", SIZES.iter().map(|n| format!("{n:>18}")).collect::<String>());
    for (name, query) in WORKLOADS {
        print!("{name:<24}");
        for &n in SIZES {
            match measure(query, n) {
                Some((ms, heap)) => print!("{:>18}", format!("{ms:.1}ms/{}K", heap / 1024)),
                None => print!("{:>18}", "unimplemented"),
            }
        }
        println!();
    }

    print!("\ndurability round-trip (commit survives reopen, rollback leaves no trace): ");
    match durability_roundtrip() {
        Ok(true) => println!("ok"),
        Ok(false) => println!("FAIL (recovered state wrong)"),
        Err(()) => println!("unimplemented"),
    }

    let _ = std::panic::take_hook();
    println!("\nprocess peak RSS: {} KiB", peak_rss_kib());
}
