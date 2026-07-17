//! `NamespaceMeta` — the SQL name and backing file of one database namespace within a
//! connection, held in a per-connection registry parallel to the pager/catalog stores.
//!
//! A connection sees several schemas (`main`, `temp`, and any `ATTACH`ed files). Each
//! has a [`DbIndex`] (`main` = 0, `temp` = 1, attached = 2..) and a SQL name a
//! schema-qualified reference (`aux.tbl`) resolves against. This record is the name +
//! backing-file half of a namespace (the pager/catalog are the other half); the engine
//! stores a `Vec<NamespaceMeta>` aligned with its pager/catalog vecs, and the read-only
//! `MultiCatalog` borrows the slice to resolve a schema qualifier to its store.

use std::path::PathBuf;

use crate::DbIndex;

/// One namespace's SQL name and backing file. `file` is the on-disk path of a file-backed
/// database, or `None` for the transient in-memory (`:memory:` / empty-path) and `temp`
/// backings — which is exactly what `PRAGMA database_list` reports as an empty `file`.
#[derive(Clone, Debug)]
pub struct NamespaceMeta {
    /// The SQL schema name (`main`, `temp`, or an ATTACH alias), preserved in the spelling
    /// it was created/attached with. Matched case-insensitively during resolution (SQL
    /// identifiers fold over ASCII), but reported verbatim by `PRAGMA database_list`.
    pub name: String,
    /// The backing file path for a file-backed database, else `None` (in-memory / temp).
    pub file: Option<PathBuf>,
}

impl NamespaceMeta {
    /// Build a namespace record.
    pub fn new(name: impl Into<String>, file: Option<PathBuf>) -> NamespaceMeta {
        NamespaceMeta { name: name.into(), file }
    }

    /// Resolve a schema qualifier to the [`DbIndex`] it names within `namespaces`, or
    /// `None` when no namespace answers to it.
    ///
    /// This is the ONE rule every resolver (the engine's DDL router and the binder's
    /// `MultiCatalog`) shares, so they agree on which store a qualifier means:
    /// * the two FIXED built-ins first — `main` → 0, `temp`/`temporary` → 1
    ///   ([`DbIndex::from_schema_name`]) — so they resolve even before an attached
    ///   registry is populated and can never be shadowed by an attached alias;
    /// * otherwise a case-insensitive scan of the ATTACHED entries (index 2..). The scan
    ///   skips indices 0/1 because the built-ins already own `main`/`temp`, and an ATTACH
    ///   alias may never be one of them (attaching as `main`/`temp` is rejected).
    pub fn resolve(namespaces: &[NamespaceMeta], schema: &str) -> Option<DbIndex> {
        if let Some(db) = DbIndex::from_schema_name(schema) {
            return Some(db);
        }
        namespaces
            .iter()
            .enumerate()
            .skip(2)
            .find(|(_, m)| m.name.eq_ignore_ascii_case(schema))
            .map(|(i, _)| DbIndex(i as u16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(attached: &[&str]) -> Vec<NamespaceMeta> {
        let mut v = vec![NamespaceMeta::new("main", None), NamespaceMeta::new("temp", None)];
        for a in attached {
            v.push(NamespaceMeta::new(*a, None));
        }
        v
    }

    #[test]
    fn resolves_builtins_before_and_without_a_registry() {
        // The two built-ins resolve from any registry shape, including a bare `main`-only
        // one (no temp/attached entries), because `from_schema_name` answers first.
        let main_only = vec![NamespaceMeta::new("main", None)];
        assert_eq!(NamespaceMeta::resolve(&main_only, "main"), Some(DbIndex::MAIN));
        assert_eq!(NamespaceMeta::resolve(&main_only, "TEMP"), Some(DbIndex::TEMP));
        assert_eq!(NamespaceMeta::resolve(&main_only, "temporary"), Some(DbIndex::TEMP));
    }

    #[test]
    fn resolves_attached_alias_case_insensitively_in_attach_order() {
        let reg = registry(&["aux", "second"]);
        assert_eq!(NamespaceMeta::resolve(&reg, "aux"), Some(DbIndex(2)));
        assert_eq!(NamespaceMeta::resolve(&reg, "AUX"), Some(DbIndex(2)));
        assert_eq!(NamespaceMeta::resolve(&reg, "second"), Some(DbIndex(3)));
    }

    #[test]
    fn unknown_alias_is_none() {
        let reg = registry(&["aux"]);
        assert_eq!(NamespaceMeta::resolve(&reg, "missing"), None);
    }

    #[test]
    fn attached_entry_named_like_a_builtin_never_shadows_it() {
        // A defensive invariant: even if an attached slot were somehow named "main", the
        // built-in mapping wins (resolve checks `from_schema_name` first), so `main`
        // always means index 0. (Attaching as `main` is rejected upstream, so this can't
        // arise in practice — the test pins the precedence anyway.)
        let mut reg = registry(&[]);
        reg.push(NamespaceMeta::new("main", None)); // index 2, pathological
        assert_eq!(NamespaceMeta::resolve(&reg, "main"), Some(DbIndex::MAIN));
    }
}
