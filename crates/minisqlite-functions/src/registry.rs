//! The function registry: name (+ argument count) -> resolved function handle.
//!
//! SQLite function names are ASCII case-insensitive, and a single name can carry
//! several implementations distinguished by argument count (e.g. `unhex(X)` vs
//! `unhex(X,Y)`) or accept a variable number of arguments (`char(...)`). The
//! registry captures both: each name maps to a list of `(Arity, handle)` entries,
//! and resolution picks the first entry whose arity accepts the call's argument
//! count. Keys are stored and looked up lowercased so `TYPEOF`, `TypeOf`, and
//! `typeof` all resolve to the same handle.
//!
//! Scalar and aggregate namespaces are kept separate so the binder can classify a
//! call (`is_aggregate`) and so resolving `count` as a scalar — or `typeof` as an
//! aggregate — fails with the same "no such function" wording real SQLite uses.

use std::collections::HashMap;
use std::sync::Arc;

use minisqlite_expr::{AggregateFunction, ScalarFunction};
use minisqlite_types::{Error, Result};

/// How many arguments a registered function accepts. A name may hold several
/// entries with different arities to model overload-by-arg-count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// Exactly `n` arguments.
    Exact(usize),
    /// Between `lo` and `hi` arguments, inclusive.
    Range(usize, usize),
    /// `n` or more arguments.
    AtLeast(usize),
    /// Any number of arguments (including zero).
    Any,
}

impl Arity {
    /// Whether a call with `argc` arguments is accepted by this arity.
    fn accepts(self, argc: usize) -> bool {
        match self {
            Arity::Exact(n) => argc == n,
            Arity::Range(lo, hi) => lo <= argc && argc <= hi,
            Arity::AtLeast(n) => argc >= n,
            Arity::Any => true,
        }
    }
}

/// A resolved registry of built-in (and, in principle, user-defined) SQL
/// functions. Resolving a name yields a cheap `Arc` handle clone; built once per
/// connection via [`FunctionRegistry::builtins`].
pub struct FunctionRegistry {
    scalars: HashMap<String, Vec<(Arity, Arc<dyn ScalarFunction>)>>,
    aggregates: HashMap<String, Vec<(Arity, Arc<dyn AggregateFunction>)>>,
}

impl FunctionRegistry {
    /// An empty registry, the starting point for [`builtins`](Self::builtins) and
    /// for tests that install a bespoke function set.
    pub fn empty() -> Self {
        FunctionRegistry { scalars: HashMap::new(), aggregates: HashMap::new() }
    }

    /// A fully-populated registry of the engine's built-in functions. Each family
    /// registers itself; families not yet implemented register nothing, so the set
    /// grows as those modules are filled in without any change here.
    pub fn builtins() -> Self {
        let mut r = Self::empty();
        crate::scalar::register(&mut r);
        crate::agg::register(&mut r);
        crate::datetime::register(&mut r);
        crate::json::register(&mut r);
        r
    }

    /// Register a scalar function under `name` (stored lowercased) for the given
    /// arity. Multiple registrations under one name model overloads.
    pub fn add_scalar(&mut self, name: &str, arity: Arity, f: Arc<dyn ScalarFunction>) {
        self.scalars.entry(name.to_ascii_lowercase()).or_default().push((arity, f));
    }

    /// Register an aggregate function under `name` (stored lowercased) for the
    /// given arity.
    pub fn add_aggregate(&mut self, name: &str, arity: Arity, f: Arc<dyn AggregateFunction>) {
        self.aggregates.entry(name.to_ascii_lowercase()).or_default().push((arity, f));
    }

    /// Resolve a scalar call of `name` with `argc` arguments to its handle.
    ///
    /// Errors use SQLite's wording so they match real sqlite: an entirely
    /// unknown name (including a name that exists only as an aggregate) yields
    /// `no such function: <name>`; a known scalar name called with an unaccepted
    /// argument count yields `wrong number of arguments to function <name>()`. The
    /// *original* spelling of `name` is used in both messages.
    pub fn resolve_scalar(&self, name: &str, argc: usize) -> Result<Arc<dyn ScalarFunction>> {
        resolve_overload(&self.scalars, name, argc)
    }

    /// Resolve an aggregate call of `name` with `argc` arguments to its handle.
    /// Mirrors [`resolve_scalar`](Self::resolve_scalar): an unknown name (including
    /// a scalar-only name) is `no such function`, and a known aggregate with a bad
    /// argument count is `wrong number of arguments`.
    pub fn resolve_aggregate(&self, name: &str, argc: usize) -> Result<Arc<dyn AggregateFunction>> {
        resolve_overload(&self.aggregates, name, argc)
    }

    /// Whether any aggregate is registered under `name` (at any arity). The binder
    /// uses this to decide whether a call is an aggregate before resolving it.
    pub fn is_aggregate(&self, name: &str) -> bool {
        self.aggregates.contains_key(&name.to_ascii_lowercase())
    }

    /// Whether `name` names any known function — scalar or aggregate, at any arity.
    pub fn is_known(&self, name: &str) -> bool {
        let key = name.to_ascii_lowercase();
        self.scalars.contains_key(&key) || self.aggregates.contains_key(&key)
    }
}

/// Shared resolution for both namespaces: case-fold `name`, then pick the first
/// overload whose arity accepts `argc`. An unknown name is `no such function`; a
/// known name with no accepting arity is `wrong number of arguments`. Scalar and
/// aggregate resolution funnel through here so their wording and overload
/// selection cannot drift apart. `T: ?Sized` lets one body serve both
/// trait-object handle types (`Arc<dyn ScalarFunction>` / `Arc<dyn AggregateFunction>`).
fn resolve_overload<T: ?Sized>(
    map: &HashMap<String, Vec<(Arity, Arc<T>)>>,
    name: &str,
    argc: usize,
) -> Result<Arc<T>> {
    match map.get(&name.to_ascii_lowercase()) {
        None => Err(no_such_function(name)),
        Some(overloads) => overloads
            .iter()
            .find(|(arity, _)| arity.accepts(argc))
            .map(|(_, f)| Arc::clone(f))
            .ok_or_else(|| wrong_arg_count(name)),
    }
}

/// SQLite's message for an unknown function name (original spelling preserved).
fn no_such_function(name: &str) -> Error {
    Error::sql(format!("no such function: {name}"))
}

/// SQLite's message for a known name called with an unaccepted argument count.
fn wrong_arg_count(name: &str) -> Error {
    Error::sql(format!("wrong number of arguments to function {name}()"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_expr::{AggregateAccumulator, FnContext};
    use minisqlite_types::{Collation, Value};

    /// A do-nothing aggregate, enough to exercise registration/resolution/
    /// classification without depending on the (later) real aggregate family.
    #[derive(Debug)]
    struct StubAgg;
    impl AggregateFunction for StubAgg {
        fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
            Box::new(StubAcc)
        }
    }
    struct StubAcc;
    impl AggregateAccumulator for StubAcc {
        fn step(&mut self, _args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
            Ok(())
        }
        fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn arity_accepts_matches_shape() {
        assert!(Arity::Exact(1).accepts(1));
        assert!(!Arity::Exact(1).accepts(0));
        assert!(!Arity::Exact(1).accepts(2));
        assert!(Arity::Range(1, 2).accepts(1));
        assert!(Arity::Range(1, 2).accepts(2));
        assert!(!Arity::Range(1, 2).accepts(0));
        assert!(!Arity::Range(1, 2).accepts(3));
        assert!(Arity::AtLeast(2).accepts(2));
        assert!(Arity::AtLeast(2).accepts(9));
        assert!(!Arity::AtLeast(2).accepts(1));
        assert!(Arity::Any.accepts(0));
        assert!(Arity::Any.accepts(100));
    }

    #[test]
    fn resolve_scalar_is_case_insensitive() {
        let reg = FunctionRegistry::builtins();
        assert!(reg.resolve_scalar("typeof", 1).is_ok());
        assert!(reg.resolve_scalar("TYPEOF", 1).is_ok());
        assert!(reg.resolve_scalar("TypeOf", 1).is_ok());
    }

    #[test]
    fn resolve_scalar_wrong_arity_message_uses_original_spelling() {
        let reg = FunctionRegistry::builtins();
        match reg.resolve_scalar("typeof", 2) {
            Err(Error::Sql(m)) => assert_eq!(m, "wrong number of arguments to function typeof()"),
            other => panic!("expected wrong-arity Sql error, got {other:?}"),
        }
        // Original (upper) spelling flows into the message verbatim.
        match reg.resolve_scalar("TYPEOF", 0) {
            Err(Error::Sql(m)) => assert_eq!(m, "wrong number of arguments to function TYPEOF()"),
            other => panic!("expected wrong-arity Sql error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_scalar_unknown_name_is_no_such_function() {
        let reg = FunctionRegistry::builtins();
        match reg.resolve_scalar("nope", 1) {
            Err(Error::Sql(m)) => assert_eq!(m, "no such function: nope"),
            other => panic!("expected no-such-function Sql error, got {other:?}"),
        }
    }

    #[test]
    fn overload_by_arg_count_picks_the_matching_arity() {
        // `unhex` is registered as Range(1,2); both 1- and 2-arg calls resolve.
        let reg = FunctionRegistry::builtins();
        assert!(reg.resolve_scalar("unhex", 1).is_ok());
        assert!(reg.resolve_scalar("unhex", 2).is_ok());
        match reg.resolve_scalar("unhex", 3) {
            Err(Error::Sql(m)) => assert_eq!(m, "wrong number of arguments to function unhex()"),
            other => panic!("expected wrong-arity Sql error, got {other:?}"),
        }
    }

    #[test]
    fn variadic_char_accepts_any_argc() {
        let reg = FunctionRegistry::builtins();
        for argc in [0usize, 1, 5, 50] {
            assert!(reg.resolve_scalar("char", argc).is_ok(), "char/{argc} should resolve");
        }
    }

    #[test]
    fn aggregate_and_scalar_namespaces_are_separate() {
        let mut reg = FunctionRegistry::empty();
        reg.add_scalar("typeof", Arity::Exact(1), Arc::new(ScalarStub));
        reg.add_aggregate("count", Arity::Range(0, 1), Arc::new(StubAgg));

        // Classification.
        assert!(reg.is_aggregate("count"));
        assert!(reg.is_aggregate("COUNT")); // case-insensitive
        assert!(!reg.is_aggregate("typeof"));
        assert!(reg.is_known("typeof"));
        assert!(reg.is_known("COUNT"));
        assert!(!reg.is_known("nope"));

        // A scalar-only name resolved as an aggregate is "no such function".
        match reg.resolve_aggregate("typeof", 1) {
            Err(Error::Sql(m)) => assert_eq!(m, "no such function: typeof"),
            other => panic!("expected no-such-function, got {other:?}"),
        }
        // An aggregate-only name resolved as a scalar is "no such function".
        match reg.resolve_scalar("count", 1) {
            Err(Error::Sql(m)) => assert_eq!(m, "no such function: count"),
            other => panic!("expected no-such-function, got {other:?}"),
        }
        // Aggregate resolves at an accepted arity, errors on a bad one.
        assert!(reg.resolve_aggregate("count", 0).is_ok());
        assert!(reg.resolve_aggregate("count", 1).is_ok());
        match reg.resolve_aggregate("count", 2) {
            Err(Error::Sql(m)) => assert_eq!(m, "wrong number of arguments to function count()"),
            other => panic!("expected wrong-arity, got {other:?}"),
        }
    }

    /// A trivial scalar used only to occupy a name in the namespace-separation test.
    #[derive(Debug)]
    struct ScalarStub;
    impl ScalarFunction for ScalarStub {
        fn call(&self, _args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
            Ok(Value::Null)
        }
    }
}
