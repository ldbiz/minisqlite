# Caveats and Unknowns

This document lists aspects of the repository that could not be fully confirmed from the code, or where uncertainty remains.

## Concurrency Model Details

**What's unclear:**

The in-process multi-connection coordination (especially for WAL mode) is documented but not extensively tested in the automated suite.

The behavior under concurrent access from separate processes is explicitly unsupported, but the failure modes (corruption vs. error) are not systematically documented.

**Questions for maintainers:**

- What are the exact failure modes when multiple processes access the same database?
- Is there a deliberate ordering of operations in the process-global WAL registry?
- Are there known race conditions that are acceptable because single-process use is the supported model?

## Statistics and Cost-Based Planning

**What's unclear:**

The planner implements a fixed selectivity ladder.

`ANALYZE` writes real `sqlite_stat1` rows, but the README states "the planner does not read them yet."

It's ambiguous whether this is a missing feature or a design choice.

**Behavior inferred but not confirmed:**

The planner may read statistics in the future, but currently all choices are structural (uniqueness, prefix length, provable orderings).

**Questions for maintainers:**

- Is cost-based planning planned for a future version?
- Are there tests for `ANALYZE` output correctness, even though it's not consumed?

## Compatibility with Real SQLite Versions

**What's unclear:**

The README demonstrates two-way file compatibility with `sqlite3`, but doesn't specify which SQLite version(s) were tested.

SQLite's file format is backward-compatible, but features evolve.

**Questions for maintainers:**

- Which SQLite version(s) is minisqlite targeting for compatibility?
- Are there known incompatibilities with newer or older SQLite versions?

## Performance Characteristics

**What's unclear:**

The benchmark harness measures time, heap, and RSS, but there are no documented performance targets or regression thresholds.

It's unclear what constitutes acceptable performance for different workload sizes.

**Questions for maintainers:**

- Are there expected performance ranges for each workload?
- Should performance changes gate merges, or are benchmarks purely informational?

## UTF-16 Encoding Support

**What's visible:**

The code supports UTF-16LE and UTF-16BE text encodings.

Format tests exist for UTF-16 databases.

**What's unclear:**

Whether UTF-16 support is fully feature-complete or if some edge cases remain untested.

The README mentions UTF-16 in the context of format compatibility but doesn't detail the level of support.

**Questions for maintainers:**

- Are there known limitations with UTF-16 databases?
- Are collations fully implemented for UTF-16?

## Reserved Bytes at End of Pages

**What's visible:**

The README mentions "databases with reserved bytes at the end of each page" as supported.

**What's unclear:**

Reserved bytes are an SQLite feature for application-specific metadata.

It's unclear if minisqlite merely preserves them or if there's API support for reading/writing them.

**Questions for maintainers:**

- Does minisqlite expose reserved bytes through any API?
- Or does it only preserve them when reading/writing existing databases?

## Differential Testing Against Real SQLite

**What's visible:**

The README states "differential testing against real SQLite happens outside this repo."

**What's unclear:**

Where this differential testing lives, how comprehensive it is, and whether it's automated.

**Questions for maintainers:**

- Is there a separate test suite that runs both engines and compares results?
- How often is it run?
- What are the known divergences?

## `WITHOUT ROWID` Table Performance

**What's visible:**

`WITHOUT ROWID` tables are implemented and tested.

**What's unclear:**

Whether the planner makes optimal use of `WITHOUT ROWID` tables (e.g., avoiding unnecessary primary key lookups).

**Questions for maintainers:**

- Are there planner optimizations specific to `WITHOUT ROWID` tables?
- Are there known performance gaps compared to regular tables?

## Trigger Recursion Limits

**What's visible:**

Triggers support recursion gated by `PRAGMA recursive_triggers`.

Recursion is bounded.

**What's unclear:**

The exact recursion limit and whether it matches SQLite's limit.

**Questions for maintainers:**

- What is the recursion limit for triggers?
- Does it match SQLite's behavior?

## Auto-Vacuum Edge Cases

**What's visible:**

Auto-vacuum is implemented with full and incremental modes.

Pointer maps are derived at commit by walking the B-tree forest.

**What's unclear:**

Edge cases around failures during auto-vacuum (e.g., partial compaction followed by crash).

**Behavior inferred:**

The code mentions that a pass that cannot verify rolls back rather than risking corruption, which is reassuring.

But the recovery path after a failed auto-vacuum is not explicitly documented.

**Questions for maintainers:**

- What happens if auto-vacuum fails mid-commit?
- Is the transaction always rolled back cleanly, or are there edge cases?

## JSON Function Completeness

**What's visible:**

The README lists JSON functions including `json_extract`, `json_set`, `json_patch`, `json_group_array`, `json_each`, and `json_tree`.

**What's unclear:**

Whether all JSON functions from SQLite 3.38+ are implemented, or if there are known gaps.

**Questions for maintainers:**

- Is the JSON function set complete compared to a specific SQLite version?
- Are there known missing JSON functions?

## Window Function Frame Edge Cases

**What's visible:**

Window functions support the full frame grammar (ROWS / RANGE / GROUPS, all EXCLUDE modes).

Frame resolution is tested against the spec's worked examples.

**What's unclear:**

Whether all combinations of frame types, bounds, and exclude modes are tested, or if some rare combinations lack coverage.

**Questions for maintainers:**

- Are there known untested frame combinations?
- Are there known edge cases where frame behavior diverges from SQLite?

## External Dependencies Policy

**What's visible:**

The workspace has exactly one external dependency: `elsa` for the page cache.

**What's unclear:**

Whether this is a strict policy (minimize dependencies) or just the current state.

**Questions for maintainers:**

- Is there a deliberate policy to avoid external dependencies?
- Under what circumstances would a new dependency be acceptable?

## Future Feature Priorities

**What's unclear:**

Which missing SQLite features (if any) are planned for future implementation.

**Questions for maintainers:**

- Are there plans to add features like full-text search (FTS)?
- Are there plans to add ATTACH DATABASE limits or other constraints?
- Is prepared-statement API planned?

## Testing on Different Platforms

**What's unclear:**

Whether the test suite runs on Windows, macOS, and Linux, or if development/testing is primarily on one platform.

File I/O and fsync behavior can vary across platforms.

**Questions for maintainers:**

- Which platforms are officially supported?
- Are there CI runs for all target platforms?

## Error Message Fidelity

**What's visible:**

The README mentions "Constraint errors carry SQLite's extended result codes and message shapes."

**What's unclear:**

Whether error messages are byte-for-byte identical to SQLite, or just structurally similar.

**Questions for maintainers:**

- Do error messages exactly match SQLite?
- Are there known cases where messages differ?
