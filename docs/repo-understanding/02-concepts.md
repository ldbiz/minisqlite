# Core Concepts

This glossary covers domain-specific concepts central to how minisqlite works.

Generic Rust, Cargo, or SQLite language features are excluded unless they play a specific architectural role in this repository.

## B-tree

**Meaning in this repo:**

The fundamental data structure for storing tables and indexes on disk.

Two shapes exist:
- **Table B-trees:** B+-like structure where rows live only in leaves, keyed by rowid
- **Index B-trees:** Classic B-trees where interior dividers are themselves index entries

**Where it appears:**

`crates/minisqlite-btree/src/`

**Related files:**

- `tree.rs` — Core structure and operations
- `insert.rs` — Insert with in-place splice or page rebuild
- `delete.rs` — Delete with balance maintenance
- `cursor.rs` — Cursor navigation for scanning and seeking
- `index*.rs` — Index-specific operations

## Pager

**Meaning in this repo:**

The storage seam that abstracts page access and transaction semantics.

The pager provides the page cache, copy-on-write transactions, savepoints, and the freelist.

All storage I/O goes through the pager trait.

**Where it appears:**

`crates/minisqlite-pager/src/pager.rs`

**Related files:**

- `cow.rs` — Copy-on-write transaction layer
- `store.rs` — Store trait for different backends (mem, disk, WAL)
- `diskstore.rs` — Rollback-journal mode backing
- `walstore.rs` — WAL mode backing
- `alloc.rs` — Freelist and page allocation policy

## Catalog

**Meaning in this repo:**

The schema management system.

The source of truth is the `sqlite_schema` B-tree on page 1, which stores one row per database object with its original `CREATE` SQL text.

The catalog parses these rows into typed definitions (`TableDef`, `IndexDef`, `ViewDef`, `TriggerDef`) and maintains a case-insensitive cache.

**Where it appears:**

`crates/minisqlite-catalog/src/`

**Related files:**

- `schemacatalog.rs` — Catalog trait and cache
- `schema_row.rs` — Reading/writing `sqlite_schema` rows
- `def.rs` — Typed schema definition structures
- `builder.rs` — Builders for creating schema objects
- `alter_data.rs` — ALTER TABLE row rewrite logic

## EvalExpr

**Meaning in this repo:**

A register-based intermediate representation for expressions.

By execution time, all names are resolved to register indices, function handles, and comparison metadata (affinity and collation).

The evaluator runs expressions against a register frame.

**Where it appears:**

`crates/minisqlite-expr/src/eval.rs`

**Related files:**

- `pattern.rs` — LIKE and GLOB pattern matching

## Plan / PlanNode

**Meaning in this repo:**

The compiled representation of a statement.

After parsing and binding, the planner produces a `Plan`: an operator tree with access-path decisions (index vs. scan) already made.

The executor never re-plans.

**Where it appears:**

`crates/minisqlite-plan/src/`

**Related files:**

- `planner.rs` — Main planner entry point
- `query_planner.rs` — SELECT planning
- `access_path.rs` — Index selection logic
- `compile/` — Statement compilers (DML, DDL, queries)
- `bind/` — Name and expression binding

## RowCursor

**Meaning in this repo:**

The pull-based iterator abstraction used by the executor.

Each operator implements the `RowCursor` trait, producing rows on demand.

This is the classic Volcano model: operators form a tree and pull from their children.

**Where it appears:**

`crates/minisqlite-exec/src/runtime.rs` (trait definition)

**Related files:**

- `executor.rs` — Executor that drains cursors
- `ops/` — All operator implementations (~30 files)

## Namespace

**Meaning in this repo:**

A database within a connection.

Every connection has:
- `main` (index 0): the primary database
- `temp` (index 1): temporary tables, created lazily
- Attached databases (index 2+): added via ATTACH

Each namespace has its own pager, catalog, and name.

**Where it appears:**

`crates/minisqlite-engine/src/namespace.rs`

**Related files:**

- `engine.rs` — Holds the vector of namespaces
- `attach.rs` — ATTACH/DETACH logic

## Seam

**Meaning in this repo:**

An architectural boundary enforced by a single trait or `pub fn`.

Each seam represents a layer in the stack: `Engine`, `Pager`, `Catalog`, `Planner`, `Executor`, and `parse`.

The architecture is mechanically enforced by tests in `seams.rs`.

**Where it appears:**

Throughout the workspace; each major crate exposes one seam.

**Related files:**

- `crates/minisqlite/tests/seams.rs` — Tests that enforce architectural rules

## Affinity

**Meaning in this repo:**

SQLite's type coercion system.

Each column has a type affinity (TEXT, NUMERIC, INTEGER, REAL, BLOB) that influences how values are stored and compared.

Affinity is decided at bind time and carried in `EvalExpr` nodes.

**Where it appears:**

`crates/minisqlite-types/src/affinity.rs`

**Related files:**

- `cast.rs` — Type coercion logic
- `compare.rs` — Comparison with affinity and collation

## Collation

**Meaning in this repo:**

The comparison rule for text values.

SQLite defines three built-in collations:
- `BINARY` — Byte-by-byte comparison
- `NOCASE` — Case-insensitive ASCII comparison
- `RTRIM` — Ignores trailing spaces

Collation is decided at bind time and affects index selection and sorting.

**Where it appears:**

`crates/minisqlite-types/src/collation.rs`

**Related files:**

- `compare.rs` — Comparison implementation

## Copy-on-write (COW)

**Meaning in this repo:**

The transaction mechanism in the pager.

A transaction is a dirty-page overlay (`HashMap<PageId, Box<[u8]>>`).

Reads consult the overlay first, then the committed store.

Commit hands dirty pages to the store; rollback drops the overlay.

**Where it appears:**

`crates/minisqlite-pager/src/cow.rs`

**Related files:**

- `pager.rs` — Pager trait that uses COW
- `store.rs` — Store backends that receive commits

## Rollback Journal

**Meaning in this repo:**

The default durability protocol.

Before modifying pages, their pre-images are written to a `-journal` file.

After fsync, the database is modified in place.

On crash, the next `open` replays the hot journal to restore the previous commit.

**Where it appears:**

`crates/minisqlite-journal/src/`

**Related files:**

- `codec.rs` — Journal format encoding/decoding
- `writer.rs` — Writing journal frames
- `recover.rs` — Hot-journal recovery

## WAL (Write-Ahead Log)

**Meaning in this repo:**

An alternative durability protocol enabled by `PRAGMA journal_mode=wal`.

Instead of modifying the database in place, commits append frames to a `-wal` file.

The database file is untouched; readers maintain snapshots.

Checkpoints later copy committed frames back into the database.

**Where it appears:**

`crates/minisqlite-wal/src/`

**Related files:**

- `codec.rs` — WAL format encoding/decoding
- `index.rs` — Frame index for page lookups
- `checkpoint.rs` — Checkpoint algorithms (PASSIVE, FULL, RESTART, TRUNCATE)
- `checksum.rs` — Cumulative checksum validation

## Savepoint

**Meaning in this repo:**

A named sub-transaction within a transaction.

Savepoints create bounded pre-image deltas inside the dirty-page overlay.

`ROLLBACK TO` restores the savepoint state without touching the store.

**Where it appears:**

`crates/minisqlite-pager/src/cow.rs` (implementation)

**Related files:**

- `crates/minisqlite-engine/src/txn.rs` — Savepoint name stack

## Overflow

**Meaning in this repo:**

A mechanism for storing large payloads that don't fit in a single B-tree cell.

Payloads above the spill threshold continue into overflow chains using SQLite's exact split formula.

**Where it appears:**

`crates/minisqlite-fileformat/src/overflow.rs`

**Related files:**

- `crates/minisqlite-btree/src/overflow_io.rs` — Overflow read/write operations
