# Behaviour Walkthroughs

This document describes key behaviours supported by minisqlite.

## 1. Query Execution with Index Selection

**Triggering action:**

Application calls `connection.query("SELECT name FROM users WHERE id = 42")`.

**Files/modules involved:**

- `crates/minisqlite-sql/src/` — Parsing
- `crates/minisqlite-plan/src/access_path.rs` — Index selection
- `crates/minisqlite-plan/src/query_planner.rs` — Query planning
- `crates/minisqlite-exec/src/executor.rs` — Execution
- `crates/minisqlite-exec/src/ops/seek_rowid.rs` — Rowid lookup
- `crates/minisqlite-btree/src/cursor.rs` — B-tree cursor
- `crates/minisqlite-pager/src/pager.rs` — Page access

**Flow:**

The SQL text is tokenized and parsed into an AST.

The binder resolves `users` to a table definition from the catalog and `id` to a column.

The planner examines the `WHERE id = 42` constraint.

If `id` is the rowid or a `PRIMARY KEY`, the planner selects a rowid seek (O(log n) instead of a scan).

If `id` has an index, the planner may select an index seek followed by a table lookup.

The planner compiles this into a `Plan` with an operator tree.

The executor builds a `RowCursor` tree and begins pulling rows.

For a rowid seek, the cursor descends the table B-tree to find the single row with rowid 42.

The B-tree layer requests pages from the pager, which returns borrowed slices from the page cache.

The row is decoded from the page bytes and projected to the requested columns.

The executor collects the row into a `QueryResult` and returns it.

**Inputs:**

- SQL query string
- Database file with the `users` table and data

**Outputs:**

- `QueryResult` with columns `["name"]` and one row

**External calls or side effects:**

- File reads to load pages from disk (cached after first access)

**Tests:**

- `crates/minisqlite/tests/conformance_select_where.rs` — WHERE clause behavior
- `crates/minisqlite-btree/tests/` — B-tree seek and scan operations
- `crates/minisqlite-plan/tests/` — Access path selection

## 2. Transaction Commit with Rollback Journal

**Triggering action:**

Application calls:
```rust
connection.execute("BEGIN")?;
connection.execute("INSERT INTO t VALUES (1), (2)")?;
connection.execute("COMMIT")?;
```

**Files/modules involved:**

- `crates/minisqlite-engine/src/txn.rs` — Transaction state management
- `crates/minisqlite-pager/src/cow.rs` — Copy-on-write overlay
- `crates/minisqlite-pager/src/diskstore.rs` — Rollback-journal backing
- `crates/minisqlite-journal/src/writer.rs` — Journal writing
- `crates/minisqlite-btree/src/insert.rs` — B-tree insert

**Flow:**

`BEGIN` marks the transaction as open in the engine's transaction state.

The `INSERT` is parsed, planned, and executed.

The executor inserts rows into the table B-tree.

B-tree modifications call `pager.page_mut()` to get mutable page access.

The pager's copy-on-write layer adds the page to the dirty-page overlay.

At `COMMIT`, the engine calls `pager.commit()`.

The pager hands the dirty pages to the `DiskStore`.

The store writes the pre-image of each dirty page to the `-journal` file.

The store fsyncs the journal (and the directory if the journal is new).

The store writes the modified pages in place to the `.db` file.

The store fsyncs the database.

The store deletes the journal file (the commit point).

The transaction completes successfully.

**Inputs:**

- SQL statements (BEGIN, INSERT, COMMIT)
- Database file

**Outputs:**

- Modified database file with new rows
- No journal file (deleted after successful commit)

**External calls or side effects:**

- File I/O: reads (to load pages), writes (journal and database), fsyncs
- Journal file creation and deletion

**Tests:**

- `crates/minisqlite/tests/conformance_transactions.rs` — Transaction semantics
- `crates/minisqlite-journal/tests/` — Journal codec and recovery
- `crates/minisqlite-pager/tests/` — Commit and rollback logic

## 3. Schema Change with ALTER TABLE

**Triggering action:**

Application calls `connection.execute("ALTER TABLE users ADD COLUMN age INTEGER")`.

**Files/modules involved:**

- `crates/minisqlite-catalog/src/alter_data.rs` — ALTER TABLE implementation
- `crates/minisqlite-catalog/src/schema_row.rs` — Reading/writing `sqlite_schema`
- `crates/minisqlite-catalog/src/def.rs` — Schema definitions
- `crates/minisqlite-sql/src/` — Parsing the CREATE and ALTER statements

**Flow:**

The `ALTER TABLE` statement is parsed.

The engine passes it to the catalog layer.

The catalog loads the current table definition from `sqlite_schema`.

For `ADD COLUMN`, the catalog rewrites the stored `CREATE TABLE` SQL to include the new column.

The catalog parses the rewritten SQL to validate it.

The catalog writes the updated row back to `sqlite_schema` within the current transaction.

The schema cache is invalidated.

The next query re-reads the schema and sees the new column.

Existing rows are not modified: SQLite's format allows adding columns without rewriting data.

The new column reads as `NULL` for existing rows unless a `DEFAULT` is specified.

**Inputs:**

- `ALTER TABLE` SQL
- Existing table in the database

**Outputs:**

- Modified `sqlite_schema` row
- Updated table definition in the schema cache

**External calls or side effects:**

- Transaction that modifies `sqlite_schema`

**Tests:**

- `crates/minisqlite/tests/conformance_alter_table.rs` — ALTER TABLE operations
- `crates/minisqlite-catalog/tests/` — Schema manipulation

## 4. WAL Checkpoint

**Triggering action:**

Application calls `connection.execute("PRAGMA wal_checkpoint(FULL)")`.

**Files/modules involved:**

- `crates/minisqlite-engine/src/pragma.rs` — PRAGMA dispatch
- `crates/minisqlite-pager/src/walstore.rs` — WAL store implementation
- `crates/minisqlite-wal/src/checkpoint.rs` — Checkpoint algorithms
- `crates/minisqlite-wal/src/index.rs` — WAL frame index

**Flow:**

The PRAGMA is parsed and dispatched to the pragma handler.

The handler calls the pager's checkpoint method with mode `FULL`.

The pager delegates to the `WalStore`.

The checkpoint algorithm determines which frames can be copied back to the database:
- Frames up to the oldest active reader's snapshot
- Only up to a commit boundary (never partial transactions)

The store copies committed frames from the `-wal` file into the `.db` file.

After copying, the store updates the WAL header to mark the checkpoint point.

In `FULL` mode, the checkpoint waits for all readers to release their snapshots, then resets the WAL to start from frame 0.

In `TRUNCATE` mode, the WAL file is truncated after reset.

The checkpoint result (frames checkpointed, frames remaining) is returned to the application.

**Inputs:**

- `PRAGMA wal_checkpoint` with mode (PASSIVE, FULL, RESTART, or TRUNCATE)
- Database in WAL mode with accumulated frames

**Outputs:**

- Updated `.db` file with checkpointed pages
- Possibly reset or truncated `-wal` file

**External calls or side effects:**

- File I/O: reads from WAL, writes to database, possible WAL truncation

**Tests:**

- `crates/minisqlite/tests/conformance_wal.rs` — WAL mode and checkpoints
- `crates/minisqlite-wal/tests/` — Checkpoint algorithms

## 5. Foreign Key Cascade on DELETE

**Triggering action:**

Application calls `connection.execute("DELETE FROM parent WHERE id = 1")` on a database where `child` has a foreign key with `ON DELETE CASCADE`.

**Files/modules involved:**

- `crates/minisqlite-exec/src/ops/delete.rs` — DELETE operator
- `crates/minisqlite-exec/src/ops/fk_check.rs` — Foreign key enforcement
- `crates/minisqlite-plan/src/compile/dml.rs` — DML planning
- `crates/minisqlite-catalog/src/def.rs` — Foreign key definitions

**Flow:**

The DELETE statement is parsed and planned.

During planning, the catalog is consulted for foreign keys referencing the `parent` table.

The planner compiles a DELETE plan that includes cascading child programs.

Execution begins by scanning for parent rows to delete (WHERE id = 1).

For each parent row, the executor invokes the foreign key cascade logic.

The cascade logic builds a DELETE program for the child table: `DELETE FROM child WHERE parent_id = 1`.

This child DELETE is executed recursively, potentially triggering further cascades.

Recursion is bounded to prevent infinite loops.

After all cascades complete, the parent row is deleted.

Indexes on both parent and child tables are updated.

The transaction commits (or rollback if a constraint fails).

**Inputs:**

- DELETE statement
- Foreign key definitions with `ON DELETE CASCADE`
- Data in parent and child tables

**Outputs:**

- Deleted parent row
- Deleted child rows (cascaded)
- Updated indexes

**External calls or side effects:**

- Transaction that modifies multiple tables

**Tests:**

- `crates/minisqlite/tests/conformance_foreign_keys.rs` — Foreign key enforcement and actions
- `crates/minisqlite-exec/tests/` — DML constraint checking
