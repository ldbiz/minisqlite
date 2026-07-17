//! Catalog — the schema seam: the one store of tables and indexes. Data-definition
//! statements write it and data-manipulation statements read it through the same
//! handle, so an index lookup always finds the table its `CREATE TABLE` made and
//! the two cannot drift into separate stores. It is a shared, borrowed view over
//! the schema, never a per-statement deep copy.
//!
//! The store is backed by the `sqlite_schema` b-tree at page 1 (the on-disk source
//! of truth) with an in-memory case-insensitive cache over it, so the same code
//! path serves an in-memory database and an on-disk file.
//!
//! This root is a thin hub. Real code lands in the submodules so each is its own
//! edit cell:
//! - `catalog` — the `Catalog` seam trait.
//! - `def` — the schema definition structs (`ColumnDef`/`TableDef`/`IndexDef`/`KeyColumn`/`ViewDef`/`TriggerDef`, plus `ForeignKeyDef`/`GeneratedColumn`).
//! - `introspect` — the shared `PRAGMA` introspection row builders (statement + `pragma_*` TVF).
//! - `schema_row` — the five-column `sqlite_schema` on-disk row codec.
//! - `builder` — the `CREATE TABLE` / `CREATE INDEX` AST -> def builders (incl. the rowid-alias rule).
//! - `alter_rewrite` — pure tokenizer-based text rewrites of stored `sql` for `ALTER TABLE`.
//! - `alter_data` — the streaming `DROP COLUMN` row-data rewrite (one row resident).
//! - `drop_checks` — pure `DROP COLUMN` eligibility checks over the parsed `CREATE TABLE`.
//! - `schemacatalog` — the unified `Catalog` implementation.
//! - `multicatalog` — the read-only composite `Catalog` over the live per-namespace stores.
//! - `trigger_target` — the resolved `CREATE TRIGGER` target verdict (`TriggerTarget`).
//! - `reserved` — the shared `sqlite_`-prefix reserved-name predicate.

mod alter_data;
mod alter_rewrite;
mod builder;
mod catalog;
mod def;
mod drop_checks;
mod introspect;
mod multicatalog;
mod reserved;
mod schema_row;
mod schemacatalog;
mod trigger_target;

pub use builder::index_ast_key_exprs;
pub use catalog::Catalog;
pub use def::{
    ColumnDef, ForeignKeyDef, GeneratedColumn, IndexDef, KeyColumn, ReferentialAction, TableDef,
    TriggerDef, TriggerTargetDb, ViewDef,
};
pub use introspect::{pragma_rows, PragmaFunction};
pub use multicatalog::MultiCatalog;
pub use reserved::is_reserved_schema_name;
pub use schema_row::SchemaRow;
pub use schemacatalog::SchemaCatalog;
pub use trigger_target::TriggerTarget;
