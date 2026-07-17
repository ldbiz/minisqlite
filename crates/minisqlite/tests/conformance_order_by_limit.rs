//! Conformance battery for `ORDER BY ... LIMIT/OFFSET` — the bounded top-k sort.
//!
//! The engine keeps only the `offset + limit` rows a full sort would place first
//! (a capacity-bounded max-heap) rather than materializing the whole input, but the
//! rows it returns MUST be byte-identical to a full stable sort then the `LIMIT`. This
//! file is the behavioral proof of that identity, transcribed from the SQLite manual in
//! `spec/sqlite-doc/`:
//!
//! - `lang_select.html` §4 (ORDER BY): rows are ordered by the left-most key, ties by
//!   the next, each `ASC` (default) / `DESC`; a NULL sorts before any value, so NULLs
//!   lead an `ASC` order and trail a `DESC` one, and `NULLS FIRST`/`NULLS LAST` override
//!   that placement.
//! - `lang_select.html` §5 (LIMIT): "the SELECT returns the first N rows"; a LIMIT
//!   larger than the row count returns all rows; `LIMIT 0` returns none; `OFFSET M`
//!   skips the first M then returns the next N.
//! - `datatype3.html` §6 (sorting): INTEGER and REAL are interspersed in numeric order
//!   and NO storage-class conversion happens before the sort — a returned value keeps
//!   the class it was stored as.
//! - `datatype3.html` §7 (collation): `NOCASE` folds the 26 ASCII uppercase letters to
//!   lower case before comparing.
//!
//! HOW THE EXPECTATIONS ARE DERIVED (independently of the engine): most cases run over a
//! deterministic 200-row dataset whose ground truth is built in this file. The expected
//! ordering is a Rust `slice::sort_by` (which is STABLE — ties keep insertion order,
//! matching SQLite's rowid-order tie behaviour) over that ground truth, NOT a value read
//! back from the engine. [`reference_matches_the_engine_full_sort`] cross-checks the
//! reference against the engine's SEPARATELY-implemented full-sort path over all 200
//! rows, so `reference == engine-full` and `engine-bounded == engine-full[..k]` jointly
//! pin `engine-bounded == reference[..k]`. The numeric/NULL cases are additionally
//! hand-derived hardcoded rows (small enough to verify by eye) for a fully independent
//! anchor.

mod conformance;

use std::cmp::Ordering;

use conformance::*;
use minisqlite::{Connection, Value};

// ---- Ground-truth dataset (built here; never read back from the engine) -------

/// One source row. `seq` is the insertion order (0..199), stored as a column so a
/// tie-broken result is directly observable: rows equal on the ORDER BY key must come
/// back in ascending `seq`.
#[derive(Clone)]
struct R {
    seq: i64,
    k: Option<i64>,
    a: i64,
    b: i64,
    t: String,
}

/// A deterministic 200-row dataset. The columns are chosen so each ordering under test
/// is unambiguous yet discriminating:
/// * `k`: a small range (0..30) with ~1-in-17 NULLs, so values repeat HEAVILY (many
///   ties → tie-stability is exercised) and the scrambled scan order differs from the
///   sorted order (a truncate-then-sort bug would return the wrong rows).
/// * `a`, `b`: `a` has 12 heavily-duplicated values; `b` is a permutation of 0..199
///   (globally distinct), so `ORDER BY a, b DESC` is fully determined (no residual ties).
/// * `t`: a mixed-case pool with case-variant duplicates, so `COLLATE NOCASE` both
///   reorders vs BINARY and creates NOCASE-equal ties whose order must follow insertion.
fn dataset() -> Vec<R> {
    let pool = ["banana", "Apple", "cherry", "Date", "apple", "Banana", "CHERRY", "date"];
    (0..200i64)
        .map(|seq| R {
            seq,
            k: if seq % 17 == 0 { None } else { Some((seq * 37 + 13) % 30) },
            a: (seq * 7) % 12,
            b: (seq * 101 + 5) % 200,
            t: pool[(seq as usize) % pool.len()].to_string(),
        })
        .collect()
}

/// Populate `db` with `data` in `seq` order, so rowid (== scan order) equals insertion
/// order and the stable-sort tie-break is observable.
fn populate(db: &mut Connection, data: &[R]) {
    exec(db, "CREATE TABLE big(seq INTEGER, k INTEGER, a INTEGER, b INTEGER, t TEXT)");
    for r in data {
        let kv = r.k.map_or_else(|| "NULL".to_string(), |v| v.to_string());
        exec(
            db,
            &format!(
                "INSERT INTO big(seq, k, a, b, t) VALUES ({}, {}, {}, {}, '{}')",
                r.seq, kv, r.a, r.b, r.t
            ),
        );
    }
}

/// A fresh in-memory db populated with the dataset, plus the dataset itself for
/// deriving expected orderings.
fn big() -> (Connection, Vec<R>) {
    let data = dataset();
    let mut db = mem();
    populate(&mut db, &data);
    (db, data)
}

// ---- Reference orderings: a STABLE sort over the ground truth (not the engine) --

/// Stable-sort a copy of `data` by `cmp`. `slice::sort_by` is stable, so rows that
/// compare `Equal` keep their input (`seq`) order — exactly SQLite's tie behaviour.
fn stable<F: FnMut(&R, &R) -> Ordering>(data: &[R], mut cmp: F) -> Vec<R> {
    let mut v = data.to_vec();
    v.sort_by(|x, y| cmp(x, y));
    v
}

fn kval(k: &Option<i64>) -> Value {
    match k {
        Some(v) => int(*v),
        None => null(),
    }
}

fn proj_k_seq(rows: &[R]) -> Vec<Vec<Value>> {
    rows.iter().map(|r| vec![kval(&r.k), int(r.seq)]).collect()
}
fn proj_a_b(rows: &[R]) -> Vec<Vec<Value>> {
    rows.iter().map(|r| vec![int(r.a), int(r.b)]).collect()
}
fn proj_t_seq(rows: &[R]) -> Vec<Vec<Value>> {
    rows.iter().map(|r| vec![text(&r.t), int(r.seq)]).collect()
}

/// `ORDER BY k` (ASC default): NULLs first, then integers ascending.
fn k_asc(x: &R, y: &R) -> Ordering {
    match (&x.k, &y.k) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(a), Some(b)) => a.cmp(b),
    }
}
/// `ORDER BY k DESC`: values descending, NULLs LAST (the DESC default placement).
fn k_desc(x: &R, y: &R) -> Ordering {
    match (&x.k, &y.k) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => b.cmp(a),
    }
}
/// `ORDER BY k NULLS LAST` (implicit ASC): values ascending, NULLs pushed to the end.
fn k_asc_nulls_last(x: &R, y: &R) -> Ordering {
    match (&x.k, &y.k) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.cmp(b),
    }
}
/// `ORDER BY a, b DESC`: `a` ascending, ties broken by `b` descending.
fn a_asc_b_desc(x: &R, y: &R) -> Ordering {
    x.a.cmp(&y.a).then_with(|| y.b.cmp(&x.b))
}
/// `ORDER BY t COLLATE NOCASE`: ASCII case-folded byte comparison.
fn t_nocase(x: &R, y: &R) -> Ordering {
    x.t.to_ascii_lowercase().as_bytes().cmp(y.t.to_ascii_lowercase().as_bytes())
}

/// Assert `bounded` equals the first `n` rows of `full`, cell-by-cell via `value_eq`
/// (`Value` has no `PartialEq`). The engine's bounded top-k path must match its own
/// full-sort path truncated to `n`.
fn assert_prefix(bounded: &[Vec<Value>], full: &[Vec<Value>], n: usize, ctx: &str) {
    let want = &full[..n.min(full.len())];
    assert_eq!(bounded.len(), want.len(), "{ctx}: row count");
    for (i, (g, e)) in bounded.iter().zip(want).enumerate() {
        assert!(
            g.len() == e.len() && g.iter().zip(e).all(|(x, y)| value_eq(x, y)),
            "{ctx}: row {i}: bounded {g:?} != full-sort-truncated {e:?}"
        );
    }
}

// ---- The reference anchor ----------------------------------------------------

#[test]
fn reference_matches_the_engine_full_sort() {
    // Cross-validate the in-test reference against the engine's (independently
    // implemented, separately tested) FULL-sort path over ALL 200 rows, for EVERY ordering
    // the bounded cases below use. If this holds, then every `engine-bounded ==
    // reference[..k]` check below is transitively pinned to the engine's full sort too
    // (reference == engine-full and engine-bounded == reference[..k] give
    // engine-bounded == engine-full[..k]) — without ever reading a bounded expectation back
    // from the engine, and self-anchoring the DESC / NULLS LAST / two-key / NOCASE cases as
    // well as the plain ASC one.
    let (mut db, data) = big();
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k", &proj_k_seq(&stable(&data, k_asc)));
    assert_rows(
        &mut db,
        "SELECT k, seq FROM big ORDER BY k DESC",
        &proj_k_seq(&stable(&data, k_desc)),
    );
    assert_rows(
        &mut db,
        "SELECT k, seq FROM big ORDER BY k NULLS LAST",
        &proj_k_seq(&stable(&data, k_asc_nulls_last)),
    );
    assert_rows(
        &mut db,
        "SELECT a, b FROM big ORDER BY a, b DESC",
        &proj_a_b(&stable(&data, a_asc_b_desc)),
    );
    assert_rows(
        &mut db,
        "SELECT t, seq FROM big ORDER BY t COLLATE NOCASE",
        &proj_t_seq(&stable(&data, t_nocase)),
    );
}

// ---- The eight required byte-identical cases (expected from the reference) -----

#[test]
fn orderby_k_limit_5() {
    let (mut db, data) = big();
    let want = proj_k_seq(&stable(&data, k_asc)[..5]);
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k LIMIT 5", &want);
}

#[test]
fn orderby_k_limit_5_offset_3() {
    // OFFSET 3, LIMIT 5 → the ordered rows at positions 3..8. The Sort retains
    // offset+limit = 8; the Limit node then skips 3 and takes 5.
    let (mut db, data) = big();
    let want = proj_k_seq(&stable(&data, k_asc)[3..8]);
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k LIMIT 5 OFFSET 3", &want);
}

#[test]
fn orderby_k_desc_limit_5() {
    let (mut db, data) = big();
    let want = proj_k_seq(&stable(&data, k_desc)[..5]);
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k DESC LIMIT 5", &want);
}

#[test]
fn orderby_k_nulls_last_limit_5() {
    // NULLS LAST must NOT pull NULLs into the top-5: the five smallest NON-NULL rows
    // come first (NULLs only appear after every value).
    let (mut db, data) = big();
    let want = proj_k_seq(&stable(&data, k_asc_nulls_last)[..5]);
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k NULLS LAST LIMIT 5", &want);
}

#[test]
fn orderby_two_key_a_asc_b_desc_limit_5() {
    let (mut db, data) = big();
    let want = proj_a_b(&stable(&data, a_asc_b_desc)[..5]);
    assert_rows(&mut db, "SELECT a, b FROM big ORDER BY a, b DESC LIMIT 5", &want);
}

#[test]
fn limit_0_returns_no_rows() {
    let (mut db, _data) = big();
    let empty: &[Vec<Value>] = &[];
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k LIMIT 0", empty);
}

#[test]
fn limit_beyond_row_count_returns_all_ordered() {
    // LIMIT 1000 over 200 rows returns every row, still fully ordered (retain caps at
    // the row count).
    let (mut db, data) = big();
    let want = proj_k_seq(&stable(&data, k_asc));
    assert_eq!(want.len(), 200, "sanity: the dataset has 200 rows");
    assert_rows(&mut db, "SELECT k, seq FROM big ORDER BY k LIMIT 1000", &want);
}

#[test]
fn orderby_text_collate_nocase_limit_5() {
    // NOCASE folds ASCII case, so the top-5 are the case-insensitively smallest texts
    // (all fold to "apple"), tie-broken by insertion order — a set BINARY would order
    // differently (it would lead with the uppercase spellings).
    let (mut db, data) = big();
    let want = proj_t_seq(&stable(&data, t_nocase)[..5]);
    assert_rows(&mut db, "SELECT t, seq FROM big ORDER BY t COLLATE NOCASE LIMIT 5", &want);
}

// ---- Tie-stability: the bounded path == the full-sort path truncated -----------

#[test]
fn tie_stability_bounded_equals_unbounded_truncated() {
    // The heart of the optimization's correctness: for every k, the bounded top-k must
    // return EXACTLY the full stable sort truncated to k — including the relative order
    // of rows that tie on the ORDER BY key. `k` has many duplicates, so this is a real
    // tie-stability check, not just a count check. Covers both directions.
    let (mut db, _data) = big();
    let full_asc = query(&mut db, "SELECT k, seq FROM big ORDER BY k").rows;
    for n in [1usize, 5, 20, 50, 137, 199, 200, 500] {
        let bounded = query(&mut db, &format!("SELECT k, seq FROM big ORDER BY k LIMIT {n}")).rows;
        assert_prefix(&bounded, &full_asc, n, &format!("ORDER BY k LIMIT {n}"));
    }
    let full_desc = query(&mut db, "SELECT k, seq FROM big ORDER BY k DESC").rows;
    for n in [1usize, 5, 20, 50] {
        let bounded =
            query(&mut db, &format!("SELECT k, seq FROM big ORDER BY k DESC LIMIT {n}")).rows;
        assert_prefix(&bounded, &full_desc, n, &format!("ORDER BY k DESC LIMIT {n}"));
    }
}

#[test]
fn tie_stability_bounded_equals_unbounded_with_offset() {
    // Same identity with an OFFSET: retain = offset + limit, and the Limit node's skip
    // then lands on the same rows the full sort + OFFSET would.
    let (mut db, _data) = big();
    let full = query(&mut db, "SELECT k, seq FROM big ORDER BY k").rows;
    for (off, lim) in [(0usize, 10usize), (3, 7), (25, 25), (100, 50)] {
        let bounded = query(
            &mut db,
            &format!("SELECT k, seq FROM big ORDER BY k LIMIT {lim} OFFSET {off}"),
        )
        .rows;
        let want: Vec<Vec<Value>> = full.iter().skip(off).take(lim).cloned().collect();
        assert_prefix(&bounded, &want, want.len(), &format!("LIMIT {lim} OFFSET {off}"));
    }
}

// ---- Numeric / NULL ordering under the bound (no key coercion) -----------------

#[test]
fn numeric_null_ordering_under_bound_preserves_class() {
    // datatype3 §6: NULL first, then INTEGER and REAL interspersed by numeric value,
    // each KEEPING its stored class (no coercion before the sort). Hand-derived: the
    // scrambled input {3, NULL, 1.5, 2, 0.5, 1} sorts to NULL, 0.5(real), 1(int),
    // 1.5(real), 2(int), 3(int); LIMIT 4 keeps the first four with classes intact.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE num(v)");
    exec(&mut db, "INSERT INTO num VALUES (3), (NULL), (1.5), (2), (0.5), (1)");
    assert_rows(
        &mut db,
        "SELECT v FROM num ORDER BY v LIMIT 4",
        &[vec![null()], vec![real(0.5)], vec![int(1)], vec![real(1.5)]],
    );
    // The bounded slice also equals the unbounded ordering truncated (the two paths
    // agree on class and numeric interspersing).
    let full = query(&mut db, "SELECT v FROM num ORDER BY v").rows;
    let bounded = query(&mut db, "SELECT v FROM num ORDER BY v LIMIT 4").rows;
    assert_prefix(&bounded, &full, 4, "num ORDER BY v LIMIT 4");
}

#[test]
fn int_real_equal_value_tie_under_bound_matches_full_sort() {
    // An INTEGER and a REAL of equal numeric value tie in ORDER BY (SQLite leaves their
    // relative order to the stable rowid order). Whatever the full-sort path does with
    // the tie, the bounded path must do IDENTICALLY — assert the two agree rather than
    // hardcoding an order the spec calls unspecified.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tievals(v)");
    exec(&mut db, "INSERT INTO tievals VALUES (2), (2.0), (5), (2)");
    let full = query(&mut db, "SELECT v FROM tievals ORDER BY v").rows;
    for n in [1usize, 2, 3] {
        let bounded = query(&mut db, &format!("SELECT v FROM tievals ORDER BY v LIMIT {n}")).rows;
        assert_prefix(&bounded, &full, n, &format!("tievals LIMIT {n}"));
    }
}
