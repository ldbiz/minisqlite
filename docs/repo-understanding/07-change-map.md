# Change Map

This document provides practical guidance for common future changes.

For each scenario, it lists the files and directories to look at first.

## If I need to change startup or database opening behavior

**Look here:**

- `crates/minisqlite-engine/src/engine.rs` — Connection state initialization, namespace setup
- `crates/minisqlite-pager/src/pager.rs` — Pager creation and initialization
- `crates/minisqlite-pager/src/diskstore.rs` — On-disk database opening, header reading
- `crates/minisqlite-pager/src/walstore.rs` — WAL mode database opening
- `crates/minisqlite-journal/src/recover.rs` — Hot-journal recovery
- `crates/minisqlite-wal/src/codec.rs` — WAL recovery and validation
- `crates/minisqlite/src/lib.rs` — Public `open()` and `open_in_memory()` constructors

**Why these files matter:**

The engine holds connection state and coordinates namespaces.

The pager abstracts storage and handles recovery on open.

The store implementations (disk and WAL) perform the actual file I/O.

**Caveats:**

Recovery is intertwined with opening.

Changing recovery logic may require changes to both journal/WAL codecs and the store implementations.

## If I need to change transaction or savepoint behavior

**Look here:**

- `crates/minisqlite-engine/src/txn.rs` — Transaction state management, BEGIN/COMMIT/ROLLBACK
- `crates/minisqlite-pager/src/cow.rs` — Copy-on-write transaction layer, savepoint deltas
- `crates/minisqlite-pager/src/pager.rs` — Pager trait methods for transaction control
- `crates/minisqlite-pager/src/diskstore.rs` — Rollback-journal commit protocol
- `crates/minisqlite-pager/src/walstore.rs` — WAL commit protocol
- `crates/minisqlite-journal/src/writer.rs` — Journal writing
- `crates/minisqlite-wal/src/codec.rs` — WAL frame writing

**Why these files matter:**

The engine layer manages transaction state and multi-namespace coordination.

The COW layer implements the dirty-page overlay and savepoint deltas.

The stores implement the commit protocols (journal vs. WAL).

**Caveats:**

Multi-namespace transactions span all attached databases.

Changing transaction semantics must preserve the guarantee that all namespaces commit or rollback together.

## If I need to change query parsing or add new SQL syntax

**Look here:**

- `crates/minisqlite-sql/src/token.rs` — Tokenizer
- `crates/minisqlite-sql/src/keyword.rs` — Keyword recognition
- `crates/minisqlite-sql/src/parser/` — Recursive-descent and Pratt parsers
- `crates/minisqlite-sql/src/ast_*.rs` — AST node definitions
- `crates/minisqlite-plan/src/bind/` — Name resolution and binding
- `crates/minisqlite-plan/src/compile/` — Statement compilers

**Why these files matter:**

The tokenizer and parser convert SQL text to AST.

The binder resolves names and lowers expressions.

The compilers turn AST into executable plans.

**Caveats:**

New syntax requires changes at all three layers: parsing, binding, and compilation.

The parser has explicit depth and width limits to prevent stack overflow.

## If I need to change query planning or add new optimizations

**Look here:**

- `crates/minisqlite-plan/src/access_path.rs` — Index selection logic
- `crates/minisqlite-plan/src/query_planner.rs` — Query planning
- `crates/minisqlite-plan/src/planner.rs` — Planner entry point
- `crates/minisqlite-plan/src/compile/select.rs` — SELECT compilation
- `crates/minisqlite-exec/src/executor.rs` — Executor that runs plans

**Why these files matter:**

Access path selection determines whether to use an index or scan.

The query planner builds the operator tree.

The executor drains the operator tree.

**Caveats:**

The planner uses a fixed selectivity ladder, not statistics-based costing.

`ANALYZE` writes `sqlite_stat1` rows, but the planner doesn't read them yet.

Adding cost-based planning would require reading and using these statistics.

## If I need to change query execution or add new operators

**Look here:**

- `crates/minisqlite-exec/src/executor.rs` — Executor entry point
- `crates/minisqlite-exec/src/runtime.rs` — RowCursor trait definition
- `crates/minisqlite-exec/src/ops/` — Operator implementations (~30 files, one per operator)

**Why these files matter:**

The executor builds and drains the operator tree.

Each operator implements `RowCursor` and pulls from its children.

**Caveats:**

Operators must follow the register convention: a scan of an N-column rowid table emits width N+1 with rowid last.

DML operators are two-phase: drain under shared borrow, then write under exclusive access.

## If I need to change DML (INSERT/UPDATE/DELETE) or constraint checking

**Look here:**

- `crates/minisqlite-exec/src/ops/insert.rs` — INSERT operator
- `crates/minisqlite-exec/src/ops/update.rs` — UPDATE operator
- `crates/minisqlite-exec/src/ops/delete.rs` — DELETE operator
- `crates/minisqlite-exec/src/ops/dml_index.rs` — Index maintenance (shared)
- `crates/minisqlite-exec/src/ops/constraint.rs` — Constraint checking
- `crates/minisqlite-exec/src/ops/fk_check.rs` — Foreign key enforcement
- `crates/minisqlite-plan/src/compile/dml.rs` — DML planning

**Why these files matter:**

DML operators implement the two-phase protocol.

Index maintenance is shared across all DML to prevent inconsistency.

Constraint checking includes NOT NULL, CHECK, UNIQUE, PK, and FK.

**Caveats:**

Foreign key cascades compile and execute child programs recursively.

ON CONFLICT policies (ABORT, FAIL, IGNORE, REPLACE, ROLLBACK) have different undo scopes.

The engine layer applies the correct rollback scope based on the policy.

## If I need to change schema management or DDL

**Look here:**

- `crates/minisqlite-catalog/src/schemacatalog.rs` — Catalog trait and cache
- `crates/minisqlite-catalog/src/schema_row.rs` — Reading/writing `sqlite_schema`
- `crates/minisqlite-catalog/src/def.rs` — Schema definition structures
- `crates/minisqlite-catalog/src/builder.rs` — Builders for CREATE statements
- `crates/minisqlite-catalog/src/alter_data.rs` — ALTER TABLE row rewrite logic
- `crates/minisqlite-engine/src/dispatch.rs` — DDL dispatch

**Why these files matter:**

The catalog manages `sqlite_schema` persistence and the schema cache.

DDL statements update the catalog within the current transaction.

ALTER TABLE rewrites stored SQL or row data depending on the operation.

**Caveats:**

The catalog parses the stored `CREATE` SQL to rebuild definitions.

Changing schema structures requires updating both the definition types and the builders.

## If I need to change B-tree operations or storage format

**Look here:**

- `crates/minisqlite-btree/src/tree.rs` — B-tree structure
- `crates/minisqlite-btree/src/insert.rs` — Insert with balance
- `crates/minisqlite-btree/src/delete.rs` — Delete with balance
- `crates/minisqlite-btree/src/cursor.rs` — Cursor navigation
- `crates/minisqlite-fileformat/src/page.rs` — Page format codec
- `crates/minisqlite-fileformat/src/serial.rs` — Record format codec
- `crates/minisqlite-pager/src/pager.rs` — Pager interface

**Why these files matter:**

The B-tree layer implements table and index operations.

The fileformat crate is the pure codec for on-disk structures.

The pager provides page access and transaction semantics.

**Caveats:**

Format changes must maintain backward compatibility with SQLite.

The format codec is tested against hand-built byte fixtures from the spec.

Breaking compatibility will fail format tests.

## If I need to change built-in functions

**Look here:**

- `crates/minisqlite-functions/src/registry.rs` — Function registry
- `crates/minisqlite-functions/src/scalar/` — Scalar functions
- `crates/minisqlite-functions/src/agg/` — Aggregate functions
- `crates/minisqlite-functions/src/datetime/` — Date/time functions
- `crates/minisqlite-functions/src/json/` — JSON functions
- `crates/minisqlite-plan/src/bind/expr.rs` — Function binding

**Why these files matter:**

The registry maps function names to implementations.

Each function is implemented in its own module.

The binder resolves function calls and validates argument counts.

**Caveats:**

Scalar and aggregate namespaces are separate.

Window functions are implemented as aggregates with special frame handling.

Special forms (COALESCE, IIF, CASE, LIKE, GLOB) are handled in the binder, not the registry.

## If I need to change PRAGMA handling

**Look here:**

- `crates/minisqlite-engine/src/pragma.rs` — PRAGMA dispatch and implementation

**Why these files matter:**

All ~24 PRAGMAs are implemented in this single file.

PRAGMAs can read/write header fields, introspect the schema, or control behavior.

**Caveats:**

Schema-qualified PRAGMAs (`PRAGMA aux.page_count`) operate on specific namespaces.

Some PRAGMAs (e.g., `journal_mode`) modify persistent state in the database file.

## If I need to change WAL or checkpoint behavior

**Look here:**

- `crates/minisqlite-wal/src/codec.rs` — WAL format encoding/decoding
- `crates/minisqlite-wal/src/index.rs` — Frame index for page lookups
- `crates/minisqlite-wal/src/checkpoint.rs` — Checkpoint algorithms
- `crates/minisqlite-pager/src/walstore.rs` — WAL store implementation
- `crates/minisqlite-engine/src/pragma.rs` — `PRAGMA wal_checkpoint`

**Why these files matter:**

The WAL codec handles frame format and checksums.

The frame index maps pages to their latest frames.

Checkpoint algorithms determine which frames to copy back.

The WAL store performs the actual file I/O.

**Caveats:**

Checkpoints are bounded by active readers (snapshot isolation).

The four checkpoint modes (PASSIVE, FULL, RESTART, TRUNCATE) have different blocking and truncation behavior.

## If I need to change tests

**Look here:**

- `crates/minisqlite/tests/seams.rs` — Architecture enforcement tests
- `crates/minisqlite/tests/conformance_*.rs` — SQLite spec conformance (110 files)
- `crates/*/tests/` — Per-crate unit tests

**Why these files matter:**

Conformance tests document expected SQLite behavior.

Format tests validate byte-level compatibility.

Architecture tests enforce layering and seam constraints.

**Caveats:**

Conformance test assertions are never weakened to pass.

A failing test signals a divergence from the SQLite spec.

Format fixtures are hand-built, not generated by the engine.
