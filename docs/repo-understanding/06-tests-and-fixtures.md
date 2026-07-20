# Tests and Fixtures

## Test Framework

Tests use Rust's built-in `#[test]` framework with no external test dependencies.

The workspace has **5,650 test functions** across all crates.

Run all tests with:
```
cargo test --workspace
```

This takes approximately 90 seconds on standard hardware.

## Test Organization

Tests are organized into three categories:

### 1. Conformance Tests

**Location:** `crates/minisqlite/tests/conformance_*.rs` (110 files)

**Purpose:**

Verify that minisqlite matches the SQLite specification.

Every expected value is transcribed from the official SQLite documentation, never from what the engine currently returns.

Each file cites its reference sections (e.g., `datatype3.html §3.4`, `windowfunctions.html §2.2`).

**Methodology:**

Assertions compare against documented behavior.

A failing assertion signals that the engine diverges from the spec.

Assertions are never weakened to pass.

**Coverage:**

- Affinity and comparison tables
- Null three-valued logic
- Collations (BINARY, NOCASE, RTRIM)
- Every join shape (INNER, LEFT, RIGHT, FULL, CROSS, NATURAL)
- Window frames (against spec's worked examples)
- Trigger semantics
- Foreign key actions (CASCADE, SET NULL, SET DEFAULT, etc.)
- Upsert (ON CONFLICT ... DO UPDATE)
- Generated columns (VIRTUAL and STORED)
- WITHOUT ROWID tables
- JSON functions
- Date/time functions
- Cross-namespace behavior (ATTACH, temp tables)

**Example files:**

- `conformance_affinity.rs` — Type affinity rules
- `conformance_aggregates.rs` — Aggregate function behavior
- `conformance_foreign_keys.rs` — FK enforcement and cascades
- `conformance_window.rs` — Window function frames
- `conformance_wal.rs` — WAL mode and checkpoints

### 2. Format and Durability Tests

**Location:** Scattered across component crates (`**/tests/`)

**Purpose:**

Check the on-disk format against hand-built byte fixtures transcribed from the file-format spec.

Every non-obvious byte is justified by a comment citing its field and offset.

**Methodology:**

Fixtures are deliberately not produced by the engine's own writers.

Tests parse fixtures and verify every field, then write the same logical content and compare byte-for-byte.

Crash recovery is tested by fabricating crash states (hot journals, torn WAL frames, half-applied checkpoints) and reopening.

**Coverage:**

- Database header fields
- B-tree page structures (interior, leaf, table, index)
- Record format (varint serial types)
- Overflow chains
- Freelist trunk pages
- Pointer maps (auto-vacuum)
- UTF-16 encodings
- Rollback journal format
- WAL frame format and checksums

**Example tests:**

- `crates/minisqlite/tests/conformance_fileformat.rs` — Format compliance
- `crates/minisqlite-journal/tests/` — Journal codec and recovery
- `crates/minisqlite-wal/tests/` — WAL codec and checksums
- `crates/minisqlite-fileformat/tests/` — Page and record codecs

### 3. Architecture Tests

**Location:** `crates/minisqlite/tests/seams.rs`

**Purpose:**

Mechanically enforce the architectural boundaries.

**Coverage:**

- Exactly one trait per named seam, in its named crate
- No orphan crates (every crate has a reverse dependency)
- No Cargo features that select behavior
- No backup files in the tree
- One build, one live code path

**Effect:**

Drift from the architecture becomes a test failure instead of a documentation issue.

## How Tests Are Run

**Run all tests:**
```
cargo test --workspace
```

**Run tests for a specific crate:**
```
cargo test -p minisqlite-btree
```

**Run a specific test file:**
```
cargo test --test conformance_affinity
```

**Run with output:**
```
cargo test -- --nocapture
```

## Important Test Files

| File | What It Tests |
|------|---------------|
| `crates/minisqlite/tests/seams.rs` | Architectural rules enforcement |
| `crates/minisqlite/tests/conformance_*.rs` | SQLite spec compliance (110 files) |
| `crates/minisqlite-pager/tests/` | Transaction commit/rollback, savepoints |
| `crates/minisqlite-btree/tests/` | B-tree insert, delete, balance, overflow |
| `crates/minisqlite-journal/tests/` | Journal codec and recovery |
| `crates/minisqlite-wal/tests/` | WAL codec, index, checkpoint |
| `crates/minisqlite-exec/tests/` | Operator behavior, constraint checking |
| `crates/minisqlite-plan/tests/` | Access path selection, query planning |

## Behaviours Documented by Tests

Tests serve as executable documentation for:

**Query execution:**
- Index selection vs. full scan
- Join strategies (hash, nested loop, index-nested-loop)
- Aggregate evaluation with and without indexes
- Window function frame resolution
- Subquery evaluation and caching

**DML:**
- INSERT with conflict policies (ABORT, FAIL, IGNORE, REPLACE, ROLLBACK)
- UPDATE with FROM clause
- DELETE with cascading foreign keys
- UPSERT with multiple ON CONFLICT clauses
- RETURNING clause behavior

**Transactions:**
- BEGIN/COMMIT/ROLLBACK semantics
- Savepoint creation and rollback
- Autocommit for statements outside explicit transactions
- Multi-namespace transaction coordination (ATTACH)
- Deferred foreign key checking at commit

**Storage:**
- Rollback-journal atomic commit protocol
- WAL append and checkpoint
- Hot-journal recovery
- Torn WAL frame detection
- Auto-vacuum page relocation
- Freelist reuse

**Schema:**
- CREATE TABLE with constraints
- CREATE INDEX (UNIQUE, partial, on expressions)
- ALTER TABLE (RENAME, ADD COLUMN, DROP COLUMN)
- DROP with dependent object checking
- Generated columns (VIRTUAL and STORED)

## Fixtures and Sample Data

Tests create databases programmatically, not from static fixtures.

**Typical pattern:**
```rust
#[test]
fn test_something() -> Result<()> {
    let mut db = Connection::open_in_memory()?;
    db.execute("CREATE TABLE t(x, y)")?;
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")?;
    
    let result = db.query("SELECT * FROM t WHERE x = 1")?;
    assert_eq!(result.rows.len(), 1);
    Ok(())
}
```

**Hand-built byte fixtures** exist for format tests:

These are stored inline in test code as byte arrays, with comments explaining each field.

Example structure:
```rust
// Page 1: database header + sqlite_schema root
let page1: &[u8] = &[
    // Header magic
    0x53, 0x51, 0x4c, 0x69, 0x74, 0x65, 0x20, 0x66, // "SQLite format 3\0"
    // ... (justified byte by byte)
];
```

These fixtures are compared byte-for-byte against engine output.

## Coverage Gaps

While the suite is extensive, some areas have limited coverage:

**Concurrency:**

Tests are single-threaded by design (Rust's test framework).

Multi-connection scenarios (WAL readers, concurrent writers) are tested manually but not extensively in the automated suite.

**Edge cases:**

Some rare combinations (e.g., auto-vacuum with UTF-16 encoding on a 512-byte page) may have fewer test cases.

**Performance:**

The benchmark harness (`cargo bench`) measures performance but is not gated on passing thresholds.

**Compatibility with real SQLite:**

Two-way file compatibility is demonstrated in the README but not systematically tested in CI.

Differential testing against real SQLite happens outside this repository.
