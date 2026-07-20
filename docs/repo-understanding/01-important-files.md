# Important Files

This document lists only high-priority files and directories that are essential to understanding how the repository works.

## Entrypoints

| File / Directory | Role | Why it matters |
|------------------|------|----------------|
| `crates/minisqlite/src/lib.rs` | Public API facade | The entire public surface: `Connection` with 4 methods. All application interaction goes through here. |
| `crates/minisqlite-engine/src/engine.rs` | Engine trait implementation | Holds per-connection state (pagers, catalogs, namespaces). This is where database connections live. |
| `crates/minisqlite-engine/src/dispatch.rs` | Statement dispatch | Routes parsed statements to the appropriate handler (DDL, DML, query, transaction, PRAGMA). |

## Configuration

| File / Directory | Role | Why it matters |
|------------------|------|----------------|
| `Cargo.toml` | Workspace definition | Declares all 14 crates and workspace-level settings (edition 2024, version, publish policy). |
| `crates/*/Cargo.toml` | Per-crate configuration | Defines dependencies between crates, enforcing the layered architecture. |

## Core Application Logic

| File / Directory | Role | Why it matters |
|------------------|------|----------------|
| `crates/minisqlite-sql/src/` | Tokenizer and parser | Converts SQL text into AST. The entry point to the query pipeline. |
| `crates/minisqlite-plan/src/` | Binder and planner | Resolves names, selects access paths, compiles AST to executable operator trees. |
| `crates/minisqlite-exec/src/executor.rs` | Query executor | Drains the operator tree and produces result rows. |
| `crates/minisqlite-exec/src/ops/` | Operator implementations | ~30 operators (scan, filter, join, aggregate, sort, etc.), one file each. |
| `crates/minisqlite-catalog/src/` | Schema management | Reads/writes `sqlite_schema`, maintains the catalog, handles ALTER TABLE rewrites. |
| `crates/minisqlite-expr/src/eval.rs` | Expression evaluator | Register-based IR for expressions. Evaluates WHERE, SELECT, computed columns, etc. |
| `crates/minisqlite-functions/src/` | Built-in functions | ~90 scalar, aggregate, window, date/time, and JSON functions. |
| `crates/minisqlite-btree/src/tree.rs` | B-tree operations | Table and index B-tree structure, insert, delete, balance, cursor navigation. |
| `crates/minisqlite-pager/src/pager.rs` | Page cache and transactions | The storage seam: abstracts page access, copy-on-write transactions, savepoints. |
| `crates/minisqlite-pager/src/cow.rs` | Copy-on-write layer | Transaction overlay (dirty pages), savepoint deltas, commit/rollback logic. |
| `crates/minisqlite-journal/src/` | Rollback journal | Rollback-journal codec, writer, and hot-journal recovery. |
| `crates/minisqlite-wal/src/` | Write-ahead log | WAL codec, frame index, checkpoint algorithms. |
| `crates/minisqlite-fileformat/src/` | On-disk format codec | Pure codec for database files: header, pages, records, varints, overflow chains. |
| `crates/minisqlite-types/src/` | Shared types | `Value`, `Error`, affinity, collation, comparison rules used across all crates. |
| `crates/minisqlite-engine/src/txn.rs` | Transaction management | BEGIN, COMMIT, ROLLBACK, savepoint name stack, multi-namespace transaction coordination. |
| `crates/minisqlite-engine/src/pragma.rs` | PRAGMA handling | ~24 PRAGMAs for introspection, configuration, and WAL control. |
| `crates/minisqlite-engine/src/namespace.rs` | Database namespaces | Manages main, temp, and attached databases; coordinates cross-database operations. |

## Tests and Fixtures

| File / Directory | Role | Why it matters |
|------------------|------|----------------|
| `crates/minisqlite/tests/seams.rs` | Architecture enforcement | Mechanically pins the architecture (trait counts, dependency direction, no orphan crates). |
| `crates/minisqlite/tests/conformance_*.rs` | Conformance test suite | 110 files testing SQLite spec compliance. Values transcribed from official docs, not from engine output. |
| `crates/minisqlite/tests/conformance/` | Shared test utilities | Helper functions and fixtures for conformance tests. |
| `crates/*/tests/` | Per-crate test suites | Unit and integration tests for individual crates. |

## Build/Deployment/Support Scripts

| File / Directory | Role | Why it matters |
|------------------|------|----------------|
| `crates/minisqlite/benches/workloads.rs` | Performance harness | No-framework benchmark (plain `fn main`) covering 5 workloads over 1k–1M rows. Measures time, heap, RSS, and durability. |
