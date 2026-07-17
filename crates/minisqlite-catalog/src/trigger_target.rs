//! `TriggerTarget` — the engine's resolved verdict about WHERE a `CREATE TRIGGER`'s
//! `ON`-clause target lives, handed to [`Catalog::create_trigger`](crate::Catalog::create_trigger).
//!
//! A trigger's target must exist and its kind (table vs view) governs which timings are
//! legal (`INSTEAD OF` is view-only; `BEFORE`/`AFTER` are table-only,
//! `lang_createtrigger.html` §3). The concrete store validates this against its OWN cache
//! for a same-store target, but a TEMP trigger on a `main`/attached object
//! (`lang_createtrigger.html` §7) has a target the temp store cannot see. So the engine —
//! which holds every namespace — resolves the target first and tells the store the verdict
//! through this type, keeping ONE `create_trigger` entry point rather than a parallel
//! cross-namespace path.

/// Where a `CREATE TRIGGER`'s resolved `ON`-target lives, RELATIVE to the store that will
/// hold the trigger row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerTarget {
    /// The target is in the SAME store as the trigger. The store validates it against its
    /// own cache (existence and kind), exactly as before namespaces existed — every
    /// non-TEMP trigger and a TEMP trigger on a temp object take this path, and the stored
    /// [`TriggerDef::target`](crate::TriggerDef::target) is
    /// [`SameStore`](crate::TriggerTargetDb::SameStore).
    SameStore,
    /// The target is in a DIFFERENT namespace (only a TEMP trigger on a `main`/attached
    /// table or view). The engine has already resolved and validated the target; `schema`
    /// is the `ON`-clause's schema qualifier as WRITTEN — `Some(name)` for a qualified
    /// `ON aux.u` (recorded as [`ForeignSchema`](crate::TriggerTargetDb::ForeignSchema) and
    /// re-resolved by name at fire time, so it survives `DETACH`), or `None` for an
    /// unqualified `ON u` (recorded as
    /// [`ForeignUnqualified`](crate::TriggerTargetDb::ForeignUnqualified) and re-resolved by
    /// search order). `is_view` carries the target's kind (a view → `INSTEAD OF` only; a
    /// table → `BEFORE`/`AFTER` only) so the store enforces timing-vs-kind without seeing
    /// the foreign object.
    Foreign { schema: Option<String>, is_view: bool },
}
