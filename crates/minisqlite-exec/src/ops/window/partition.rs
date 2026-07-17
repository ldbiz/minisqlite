//! Partition / order / peer-group machinery for the window operator.
//!
//! One [`Partition`] holds one window partition's input ROW INDICES in window order
//! (never a copy of a row) together with each ordered row's evaluated `ORDER BY` key
//! values, and it precomputes the PEER GROUPS — maximal runs of rows whose `ORDER BY`
//! keys compare `Equal`. The frame engine ([`super::frame`]) and the ranking
//! ([`super::ranking`]) and navigation ([`super::navigation`]) window kinds are all
//! expressed against this one abstraction, so the definition of "peer" (and of "no
//! `ORDER BY` ⇒ one peer group over the whole partition") lives here once and cannot
//! drift between them.
//!
//! # Internal API (the frame engine, ranking, and navigation kinds bind against this)
//!
//! Construct with [`Partition::new`], passing `(input_index, order_key_values)` pairs
//! for every row of one partition plus the window `ORDER BY` keys; it STABLE-sorts the
//! pairs by [`compare_sort_keys`] (so equal-key rows keep input order within a group)
//! and derives the peer groups. Then:
//!
//! * [`Partition::len`] / [`Partition::is_empty`] — ordered row count.
//! * [`Partition::input_index`]`(pos)` — the input row index at ordered position `pos`.
//! * [`Partition::order_key`]`(pos)` — that row's `ORDER BY` key values (for RANGE
//!   bounds and peer detection).
//! * [`Partition::peer_bounds`]`(pos)` — the half-open `[lo, hi)` range in ordered
//!   space of the peer group containing `pos`. No `ORDER BY` ⇒ the whole partition is
//!   one peer group.
//! * [`Partition::group_index`]`(pos)` / [`Partition::group_bounds`]`(g)` /
//!   [`Partition::num_groups`] — peer groups addressed by index, in window order, for
//!   the GROUPS frame type (counting peer groups relative to the current group).
//!
//! COLLATION: `PARTITION BY` grouping is the *caller's* concern (it keys partitions under
//! each key's §7.1 collation via `WindowFunc.partition_collations` before handing rows
//! here already grouped into a partition), so this module only orders WITHIN a partition —
//! and that window `ORDER BY` DOES honor each `SortKey.collation` through
//! [`compare_sort_keys`].

use std::cmp::Ordering;

use minisqlite_expr::SortKey;
use minisqlite_types::Value;

use crate::ops::sortkey::compare_sort_keys;

/// One window partition's rows in window order, with precomputed peer groups.
///
/// Holds only row INDICES and the (small, query-bounded) `ORDER BY` key values — never
/// a copied input row. Peer groups are computed once at construction and addressed both
/// by containing position ([`Partition::peer_bounds`]) and by group index
/// ([`Partition::group_bounds`]).
pub(crate) struct Partition {
    /// Input row indices in window (`ORDER BY`) order.
    ordered: Vec<usize>,
    /// `order_vals[pos]` = the evaluated `ORDER BY` key values for ordered position
    /// `pos` (one per `ORDER BY` term; empty when there is no `ORDER BY`).
    order_vals: Vec<Vec<Value>>,
    /// Peer groups as half-open `[lo, hi)` ranges over ordered positions, in window
    /// order and contiguous (`groups[0].0 == 0`, `groups[last].1 == len`).
    groups: Vec<(usize, usize)>,
    /// `group_of[pos]` = index into `groups` of the peer group containing `pos`.
    group_of: Vec<usize>,
}

impl Partition {
    /// Build a partition from `(input_index, order_key_values)` pairs — one per row of
    /// a single partition — and the window `ORDER BY` keys.
    ///
    /// The pairs are STABLE-sorted by [`compare_sort_keys`] (so a peer group keeps input
    /// order), then split into the ordered index/key vectors, and the peer groups are
    /// derived by scanning adjacent keys: a new group begins wherever adjacent keys are
    /// not `Equal`. With no `ORDER BY` (`keys` empty) every comparison is `Equal`, so the
    /// whole partition is one peer group — SQLite's "all rows are peers" rule.
    pub(crate) fn new(mut rows: Vec<(usize, Vec<Value>)>, keys: &[SortKey]) -> Partition {
        // `slice::sort_by` is a STABLE sort, which is required so that rows with equal
        // `ORDER BY` keys (peers) retain their input order within the group — the
        // deterministic order SQLite's peer-order-dependent output (e.g. `group_concat`
        // over the default frame) is compared against.
        rows.sort_by(|a, b| compare_sort_keys(&a.1, &b.1, keys));

        let n = rows.len();
        let mut ordered = Vec::with_capacity(n);
        let mut order_vals = Vec::with_capacity(n);
        for (idx, key_vals) in rows {
            ordered.push(idx);
            order_vals.push(key_vals);
        }

        // Peer groups: a maximal run of adjacent-equal keys. Adjacent comparison is
        // sufficient because the list is sorted and key equality is transitive, so an
        // adjacent-equal pair implies the whole run is equal.
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut group_of = vec![0usize; n];
        if n > 0 {
            let mut lo = 0usize;
            for p in 1..n {
                if compare_sort_keys(&order_vals[p - 1], &order_vals[p], keys) != Ordering::Equal {
                    groups.push((lo, p));
                    lo = p;
                }
            }
            groups.push((lo, n));
            for (gi, &(a, b)) in groups.iter().enumerate() {
                for slot in &mut group_of[a..b] {
                    *slot = gi;
                }
            }
        }

        Partition { ordered, order_vals, groups, group_of }
    }

    /// Number of rows in the partition.
    pub(crate) fn len(&self) -> usize {
        self.ordered.len()
    }

    /// Whether the partition has no rows. (Paired with [`Partition::len`] for the
    /// `clippy::len_without_is_empty` lint; currently exercised only by the unit tests —
    /// neither `super::ranking` nor `super::navigation` calls it — hence `allow(dead_code)`.)
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }

    /// The input row index at ordered position `pos`.
    ///
    /// Panics only on an out-of-range `pos` (a caller bug); every caller iterates
    /// `0..len()` or a frame's positions, which are all `< len()`.
    pub(crate) fn input_index(&self, pos: usize) -> usize {
        self.ordered[pos]
    }

    /// The `ORDER BY` key values at ordered position `pos` (empty with no `ORDER BY`).
    pub(crate) fn order_key(&self, pos: usize) -> &[Value] {
        &self.order_vals[pos]
    }

    /// The half-open `[lo, hi)` ordered-position range of the peer group containing
    /// `pos`. With no `ORDER BY` this is `[0, len())` — the whole partition.
    pub(crate) fn peer_bounds(&self, pos: usize) -> (usize, usize) {
        self.groups[self.group_of[pos]]
    }

    /// The peer-group index (in window order) of ordered position `pos`.
    pub(crate) fn group_index(&self, pos: usize) -> usize {
        self.group_of[pos]
    }

    /// The half-open `[lo, hi)` ordered-position range of peer group `g`.
    pub(crate) fn group_bounds(&self, g: usize) -> (usize, usize) {
        self.groups[g]
    }

    /// The number of peer groups (≥ 1 for a non-empty partition; 0 when empty).
    pub(crate) fn num_groups(&self) -> usize {
        self.groups.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_types::Collation;

    /// An ascending `SortKey` over one column (the comparator reads the precomputed key
    /// values, so the `expr` is a placeholder).
    fn asc() -> SortKey {
        SortKey {
            expr: minisqlite_expr::EvalExpr::Column(0),
            desc: false,
            nulls_first: None,
            collation: Collation::Binary,
        }
    }

    fn desc() -> SortKey {
        SortKey {
            expr: minisqlite_expr::EvalExpr::Column(0),
            desc: true,
            nulls_first: None,
            collation: Collation::Binary,
        }
    }

    fn i(n: i64) -> Value {
        Value::Integer(n)
    }

    #[test]
    fn no_order_by_is_one_peer_group_in_input_order() {
        // No ORDER BY keys: the stable sort is the identity, and the whole partition is
        // one peer group. `ordered` keeps the input indices in their given order.
        let rows = vec![(5, vec![]), (2, vec![]), (9, vec![])];
        let p = Partition::new(rows, &[]);
        assert_eq!(p.len(), 3);
        assert_eq!((p.input_index(0), p.input_index(1), p.input_index(2)), (5, 2, 9));
        assert_eq!(p.num_groups(), 1);
        assert_eq!(p.peer_bounds(0), (0, 3));
        assert_eq!(p.peer_bounds(2), (0, 3));
    }

    #[test]
    fn empty_partition_has_no_groups() {
        let p = Partition::new(Vec::new(), &[asc()]);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.num_groups(), 0);
    }

    #[test]
    fn stable_sort_orders_by_key_and_keeps_input_order_within_peers() {
        // Input (index, key): (0,20),(1,10),(2,20),(3,10). Sorted asc by key:
        // key10 rows keep input order (1 then 3), key20 rows keep input order (0 then 2).
        let rows = vec![(0, vec![i(20)]), (1, vec![i(10)]), (2, vec![i(20)]), (3, vec![i(10)])];
        let p = Partition::new(rows, &[asc()]);
        assert_eq!(
            (0..p.len()).map(|pos| p.input_index(pos)).collect::<Vec<_>>(),
            vec![1, 3, 0, 2],
        );
        // Two peer groups: [0,2) = the key-10 rows, [2,4) = the key-20 rows.
        assert_eq!(p.num_groups(), 2);
        assert_eq!(p.group_bounds(0), (0, 2));
        assert_eq!(p.group_bounds(1), (2, 4));
        assert_eq!(p.peer_bounds(0), (0, 2));
        assert_eq!(p.peer_bounds(3), (2, 4));
        assert_eq!(p.group_index(3), 1);
    }

    #[test]
    fn desc_order_reverses_groups() {
        let rows = vec![(0, vec![i(1)]), (1, vec![i(2)]), (2, vec![i(2)]), (3, vec![i(3)])];
        let p = Partition::new(rows, &[desc()]);
        // Sorted desc: 3, then the two 2s (input order 1,2), then 1.
        assert_eq!(
            (0..p.len()).map(|pos| p.input_index(pos)).collect::<Vec<_>>(),
            vec![3, 1, 2, 0],
        );
        assert_eq!(p.num_groups(), 3);
        assert_eq!(p.group_bounds(1), (1, 3)); // the peer group of the two 2s
        assert_eq!(p.peer_bounds(1), (1, 3));
        assert_eq!(p.peer_bounds(2), (1, 3));
    }

    #[test]
    fn peer_groups_cover_every_position_contiguously() {
        // A standing invariant: groups tile [0, len) with no gaps/overlaps, and
        // group_of agrees with group_bounds for every position.
        let rows =
            vec![(0, vec![i(1)]), (1, vec![i(1)]), (2, vec![i(2)]), (3, vec![i(3)]), (4, vec![i(3)])];
        let p = Partition::new(rows, &[asc()]);
        let mut expected_lo = 0;
        for g in 0..p.num_groups() {
            let (lo, hi) = p.group_bounds(g);
            assert_eq!(lo, expected_lo, "groups must be contiguous");
            assert!(hi > lo, "a group is non-empty");
            for pos in lo..hi {
                assert_eq!(p.group_index(pos), g);
                assert_eq!(p.peer_bounds(pos), (lo, hi));
            }
            expected_lo = hi;
        }
        assert_eq!(expected_lo, p.len(), "groups tile the whole partition");
    }

    #[test]
    fn order_key_is_readable_per_position() {
        let rows = vec![(0, vec![i(30)]), (1, vec![i(10)])];
        let p = Partition::new(rows, &[asc()]);
        assert!(matches!(p.order_key(0), [Value::Integer(10)]));
        assert!(matches!(p.order_key(1), [Value::Integer(30)]));
    }
}
