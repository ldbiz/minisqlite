# Configuration and Environment

## Config Files

minisqlite has no external configuration files.

All configuration is embedded in the workspace and crate `Cargo.toml` files.

| File | Purpose | Key Settings |
|------|---------|--------------|
| `Cargo.toml` (root) | Workspace definition | Declares 14 member crates, sets edition 2024, version 0.0.0, publish = false |
| `crates/*/Cargo.toml` | Per-crate config | Defines crate dependencies, enforces layered architecture |

## Environment Variables

minisqlite does not read any environment variables.

The library has no environment-dependent behavior.

## Runtime Configuration

Runtime behavior is controlled entirely through the API and SQL:

### Connection Mode

**On-disk vs. in-memory:**
- `Connection::open(path)` — On-disk database
- `Connection::open_in_memory()` — In-memory database

Default: None. The application must explicitly choose.

Effect: In-memory databases are transient, faster (no I/O), and live only for the connection's lifetime.

### Journal Mode

**Controlled by:** `PRAGMA journal_mode = wal` or `PRAGMA journal_mode = delete`

Modes:
- `delete` (default) — Rollback-journal mode. Journal is deleted after commit.
- `wal` — Write-ahead log mode. Commits append to WAL, checkpoint copies back to database.

Effect:
- Rollback mode: Simple, single-writer. Journal file appears during transactions.
- WAL mode: Allows snapshot-isolated reads during writes. Requires periodic checkpoints.

Setting: Stored in the database file header. Takes effect for connections opened after the PRAGMA is executed.

### Page Size

**Controlled by:** `PRAGMA page_size = N` (before first write to a new database)

Valid range: 512 to 65536 bytes, must be a power of 2.

Default: 4096 bytes.

Effect: Determines the granularity of I/O and B-tree node size. Cannot be changed after the database is created (without VACUUM).

### Auto-Vacuum Mode

**Controlled by:** `PRAGMA auto_vacuum = none|full|incremental`

Modes:
- `none` (default) — Database file never shrinks automatically.
- `full` — Database compacts at every commit, reclaiming tail pages.
- `incremental` — Application controls reclamation with `PRAGMA incremental_vacuum(N)`.

Effect: Determines whether freed pages are reclaimed immediately, on demand, or never.

Setting: Stored in the database file header.

### Foreign Keys

**Controlled by:** `PRAGMA foreign_keys = on|off`

Default: Off (for SQLite compatibility).

Effect: When on, foreign key constraints are enforced. When off, they are checked but not enforced.

Setting: Per-connection, not stored in the database.

### Recursive Triggers

**Controlled by:** `PRAGMA recursive_triggers = on|off`

Default: Off.

Effect: When on, triggers can recursively fire other triggers. When off, recursion is bounded to one level.

Setting: Per-connection, not stored in the database.

## Database Header Fields

The first 100 bytes of page 1 contain the database header.

PRAGMAs can read and write these fields:

| Field | PRAGMA | Purpose |
|-------|--------|---------|
| Page size | `page_size` | Sets the page size (before first write) |
| Encoding | `encoding` | Text encoding (UTF-8, UTF-16LE, UTF-16BE) |
| User version | `user_version` | Application-defined version number |
| Application ID | `application_id` | Application-defined identifier |
| Journal mode | `journal_mode` | Rollback or WAL |
| Auto-vacuum | `auto_vacuum` | none, full, or incremental |
| Schema version | `schema_version` | Incremented on schema changes |

These are persistent and stored in the database file.

## Secrets and Credentials

minisqlite does not handle authentication or encryption.

The library operates on local files with no network access.

Applications are responsible for file-system permissions and any encryption layers (e.g., encrypted file systems).

## Local/Dev/Test Examples

There are no separate development or test configurations.

The test suite creates temporary databases programmatically:

```rust
let mut db = Connection::open_in_memory()?;
db.execute("CREATE TABLE t(x)")?;
```

On-disk tests use temporary files:

```rust
let path = PathBuf::from("/tmp/test.db");
let mut db = Connection::open(&path)?;
// ... test operations ...
std::fs::remove_file(&path)?;
```

## Build Configuration

**Standard build:**
```
cargo build --workspace
```

**Release build:**
```
cargo build --workspace --release
```

No Cargo features, no conditional compilation, no build-time switches.

The build is uniform across all environments.

## Runtime Consequences of Settings

| Setting | Consequence |
|---------|-------------|
| Journal mode = WAL | Enables snapshot-isolated reads. Readers don't block writers. WAL file grows until checkpoint. |
| Journal mode = delete | Simple single-writer model. Journal file appears during transactions. |
| Page size = 512 | More I/O operations, smaller memory footprint, smaller B-tree nodes. |
| Page size = 65536 | Fewer I/O operations, larger memory footprint, larger B-tree nodes. |
| Auto-vacuum = full | Database shrinks after deletes, but commits are slower. |
| Auto-vacuum = none | Database never shrinks, but commits are faster. |
| Foreign keys = on | Constraint violations fail the statement. Cascade actions execute. |
| Foreign keys = off | Constraints are ignored (for compatibility). |
