# Repository Summary

## Purpose

minisqlite is a from-scratch reimplementation of SQLite in Rust.

It implements the complete SQLite stack: SQL dialect, query planner, executor, transaction system, and storage engine, down to the official on-disk file format.

The library opens database files written by `sqlite3` and writes files that `sqlite3` can read back.

## Technology Stack

- **Language:** Rust (edition 2024)
- **Workspace structure:** Cargo workspace with 14 crates
- **External dependencies:** 1 (`elsa` for the page cache)
- **No `unsafe` code** in library code
- **Test coverage:** 5,650 tests

## Main Runtime Model

minisqlite is a library-only implementation with no CLI or C API.

The public API is deliberately minimal: one type (`Connection`) with four methods.

Applications link minisqlite as a Rust library and interact through this single-type facade.

## Main External Services or Dependencies

- **File system:** Reads and writes database files (`.db`, `-wal`, `-journal`)
- **elsa crate:** Provides `FrozenMap` for the append-only page cache with stable addresses

No network services, no external databases, no system SQLite dependency.

## Main Entry Points

| Entry Point | Role |
|-------------|------|
| `crates/minisqlite/src/lib.rs` | The public facade exposing `Connection` |
| `Connection::open(path)` | Opens or creates an on-disk database |
| `Connection::open_in_memory()` | Creates a transient in-memory database |
| `Connection::execute(sql)` | Runs DDL/DML statements that return no rows |
| `Connection::query(sql)` | Runs SELECT queries and returns result sets |

## Architecture Summary

The workspace is organized into 14 crates with strict layering.

Each crate owns a single concern and exposes it through one seam (a trait or `pub fn`).

**Top layer:** The `minisqlite` crate is a thin facade that re-exports types and delegates to the engine.

**Engine layer:** `minisqlite-engine` holds connection state, dispatches statements, manages transactions, handles PRAGMAs, and coordinates multiple database namespaces (main, temp, attached).

**Query pipeline:**
- `minisqlite-sql` tokenizes and parses SQL into an AST
- `minisqlite-plan` resolves names, binds expressions, selects access paths, compiles to an operator tree
- `minisqlite-exec` executes the operator tree using pull-based iterators (Volcano model)

**Schema and functions:**
- `minisqlite-catalog` manages `sqlite_schema` persistence and typed schema definitions
- `minisqlite-expr` provides a register-based expression IR and evaluator
- `minisqlite-functions` implements ~90 built-in scalar, aggregate, window, date/time, and JSON functions

**Storage stack:**
- `minisqlite-btree` implements table and index B-trees (insert, delete, balance, cursors)
- `minisqlite-pager` provides the page cache, copy-on-write transactions, savepoints, and allocation
- `minisqlite-journal` handles rollback-journal format and hot-journal recovery
- `minisqlite-wal` implements WAL format, frame index, and checkpoint algorithms
- `minisqlite-fileformat` is the pure codec for on-disk format (pages, records, varints, overflow)

**Shared types:** `minisqlite-types` defines `Value`, `Error`, affinity, collation, and comparison rules.

## Read These First

| Path | Why |
|------|-----|
| `README.md` | Comprehensive overview, architecture diagram, implemented features, testing methodology |
| `crates/minisqlite/src/lib.rs` | The entire public API (52 lines) |
| `crates/minisqlite-engine/src/engine.rs` | Connection state and the main engine trait implementation |
| `crates/minisqlite-engine/src/dispatch.rs` | Statement dispatch logic (DDL, DML, queries, transactions) |
| `crates/minisqlite-pager/src/pager.rs` | The storage seam: page cache and transaction interface |
| `crates/minisqlite-btree/src/tree.rs` | B-tree structure and operations |
| `crates/minisqlite-fileformat/src/` | On-disk format codec (header, pages, records) |
| `crates/minisqlite/tests/seams.rs` | Architectural enforcement tests |
| `Cargo.toml` | Workspace structure |
