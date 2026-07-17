//! `DbIndex` — a database-namespace index within one connection.
//!
//! A single SQLite connection sees several *schemas* (databases): `main` (the file
//! or in-memory backing), `temp` (the transient per-connection schema that holds
//! `CREATE TEMP` objects), and any `ATTACH`ed files. Each has its own b-tree store
//! (its own [`Pager`](../../minisqlite_pager)) and its own schema cache. A plan node
//! that reaches a base table therefore names WHICH namespace the table lives in, so
//! the executor opens the cursor on the right store; this newtype is that name.
//!
//! The index is positional and STABLE for the two fixed schemas: `main` is always
//! index 0 and `temp` (when it exists) is always index 1; `ATTACH`ed databases take
//! 2.. in attach order. Keeping `main` at 0 makes a database with no temp/attached
//! schema behave exactly as a single-store engine — the common hot path is unchanged.

/// The index of a database namespace within one connection's schema registry.
///
/// A small `u16` newtype (not a bare integer) so a namespace index cannot be confused
/// with a column register, a page id, or any other count the engine passes around —
/// the compiler rejects the mix. `MAIN` is always `0` and `TEMP` always `1`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DbIndex(pub u16);

impl DbIndex {
    /// The `main` database — the file (or in-memory) backing. Always index 0, so a
    /// single-store engine and the `main` slot of a multi-store one are the same place.
    pub const MAIN: DbIndex = DbIndex(0);

    /// The `temp` database — the transient, per-connection schema that holds
    /// `CREATE TEMP`/`CREATE TEMPORARY` objects. Always index 1 (created lazily on the
    /// first temp object); never written to the `main` file.
    pub const TEMP: DbIndex = DbIndex(1);

    /// This namespace's index as a `usize`, for indexing the parallel pager/catalog
    /// registries the engine keeps.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The namespace a SQL schema qualifier names, for the two FIXED schemas: `main`
    /// (case-insensitive) → [`MAIN`](DbIndex::MAIN); `temp` or `temporary` → [`TEMP`](DbIndex::TEMP).
    /// Any other qualifier is `None` — an unknown/unattached database, which the caller
    /// reports as "no such table". (An ATTACHed database's name → index is a dynamic
    /// lookup that layers on top of this pure mapping; only the two built-ins are fixed.)
    pub fn from_schema_name(schema: &str) -> Option<DbIndex> {
        if schema.eq_ignore_ascii_case("main") {
            Some(DbIndex::MAIN)
        } else if schema.eq_ignore_ascii_case("temp") || schema.eq_ignore_ascii_case("temporary") {
            Some(DbIndex::TEMP)
        } else {
            None
        }
    }
}

/// `main` is the default namespace, so a plan node built without an explicit schema
/// qualifier targets the file backing (the single-store behavior) until the binder
/// resolves a temp/attached name.
impl Default for DbIndex {
    fn default() -> Self {
        DbIndex::MAIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_indices_and_default() {
        assert_eq!(DbIndex::MAIN.index(), 0);
        assert_eq!(DbIndex::TEMP.index(), 1);
        assert_eq!(DbIndex::default(), DbIndex::MAIN);
    }

    #[test]
    fn schema_name_maps_the_two_builtins_case_insensitively() {
        assert_eq!(DbIndex::from_schema_name("main"), Some(DbIndex::MAIN));
        assert_eq!(DbIndex::from_schema_name("MAIN"), Some(DbIndex::MAIN));
        assert_eq!(DbIndex::from_schema_name("temp"), Some(DbIndex::TEMP));
        assert_eq!(DbIndex::from_schema_name("TEMP"), Some(DbIndex::TEMP));
        // `temporary` is an accepted spelling of the temp schema qualifier.
        assert_eq!(DbIndex::from_schema_name("temporary"), Some(DbIndex::TEMP));
        assert_eq!(DbIndex::from_schema_name("Temporary"), Some(DbIndex::TEMP));
    }

    #[test]
    fn unknown_schema_name_is_none() {
        // An unattached / misspelled database name is not one of the two built-ins;
        // the caller turns this into "no such table: x.y".
        assert_eq!(DbIndex::from_schema_name("aux"), None);
        assert_eq!(DbIndex::from_schema_name(""), None);
        assert_eq!(DbIndex::from_schema_name("tempo"), None);
    }
}
