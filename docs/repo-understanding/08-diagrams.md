# Diagrams

This document contains Mermaid diagrams illustrating the architecture and behavior of minisqlite.

## 1. Crate Dependency Graph

This diagram shows the 14 crates and their dependencies, enforcing the layered architecture.

```mermaid
graph TD
    APP(["Application Code"]) --> MINISQLITE

    MINISQLITE["minisqlite<br/>(facade)"]
    ENGINE["minisqlite-engine<br/>(dispatch, transactions, namespaces)"]
    SQL["minisqlite-sql<br/>(parser)"]
    PLAN["minisqlite-plan<br/>(binder, planner)"]
    EXEC["minisqlite-exec<br/>(executor)"]
    CATALOG["minisqlite-catalog<br/>(schema)"]
    EXPR["minisqlite-expr<br/>(expression IR)"]
    FUNCS["minisqlite-functions<br/>(built-ins)"]
    BTREE["minisqlite-btree<br/>(B-trees)"]
    PAGER["minisqlite-pager<br/>(page cache, transactions)"]
    JOURNAL["minisqlite-journal<br/>(rollback journal)"]
    WAL["minisqlite-wal<br/>(write-ahead log)"]
    FMT["minisqlite-fileformat<br/>(on-disk format codec)"]
    TYPES["minisqlite-types<br/>(Value, Error, affinity, collation)"]

    MINISQLITE --> ENGINE
    ENGINE --> SQL
    ENGINE --> PLAN
    ENGINE --> EXEC
    ENGINE --> CATALOG
    ENGINE --> PAGER
    PLAN --> SQL
    PLAN --> CATALOG
    PLAN --> EXPR
    PLAN --> FUNCS
    EXEC --> PLAN
    EXEC --> CATALOG
    EXEC --> EXPR
    EXEC --> FUNCS
    EXEC --> BTREE
    CATALOG --> SQL
    CATALOG --> BTREE
    BTREE --> PAGER
    PAGER --> JOURNAL
    PAGER --> WAL
    PAGER --> FMT
    JOURNAL --> FMT
    WAL --> FMT
    
    ENGINE -.-> TYPES
    SQL -.-> TYPES
    PLAN -.-> TYPES
    EXEC -.-> TYPES
    CATALOG -.-> TYPES
    EXPR -.-> TYPES
    FUNCS -.-> TYPES
    BTREE -.-> TYPES
    PAGER -.-> TYPES
    FMT -.-> TYPES
    
    style MINISQLITE fill:#e1f5ff
    style ENGINE fill:#fff4e1
    style TYPES fill:#f0f0f0
```

**Legend:**
- Solid arrows: Direct crate dependencies (in Cargo.toml)
- Dashed arrows: Dependency on `minisqlite-types` (omitted from most diagrams for clarity)

## 2. Statement Execution Flow

This diagram shows the path of a SQL statement from text to result.

```mermaid
graph LR
    SQL["SQL Text"] -->|parse| AST["Abstract Syntax Tree"]
    AST -->|dispatch| ROUTE{Statement Type?}
    
    ROUTE -->|DDL| DDL_HANDLER["DDL Handler<br/>(CREATE, DROP, ALTER)"]
    ROUTE -->|Transaction| TXN_HANDLER["Transaction Handler<br/>(BEGIN, COMMIT, etc.)"]
    ROUTE -->|PRAGMA| PRAGMA_HANDLER["PRAGMA Handler"]
    ROUTE -->|DML/Query| PLANNER["Planner"]
    
    DDL_HANDLER --> CATALOG_UPDATE["Update Catalog"]
    CATALOG_UPDATE --> COMMIT
    
    PLANNER -->|bind names| BINDING["Name Resolution"]
    BINDING -->|select paths| ACCESS["Access Path Selection"]
    ACCESS -->|compile| PLAN["Plan (Operator Tree)"]
    
    PLAN --> EXECUTOR["Executor"]
    EXECUTOR --> CURSOR["RowCursor Tree"]
    CURSOR -->|pull rows| BTREE_OPS["B-tree Operations"]
    BTREE_OPS --> PAGES["Page Access"]
    
    PAGES --> RESULT["Result Rows"]
    TXN_HANDLER --> COMMIT{Commit?}
    PRAGMA_HANDLER --> DONE
    RESULT --> COMMIT
    
    COMMIT -->|Yes| WRITE["Write Dirty Pages"]
    COMMIT -->|No| DONE["Done"]
    WRITE --> FSYNC["Fsync"]
    FSYNC --> DONE
    
    style SQL fill:#e1f5ff
    style RESULT fill:#e1ffe1
    style DONE fill:#ffe1e1
```

## 3. Query Planning and Execution Sequence

This diagram shows the detailed steps for planning and executing a SELECT query.

```mermaid
sequenceDiagram
    participant App as Application
    participant Eng as Engine
    participant Parse as Parser
    participant Plan as Planner
    participant Exec as Executor
    participant BTree as B-tree
    participant Pager as Pager

    App->>Eng: query("SELECT * FROM t WHERE x = 1")
    Eng->>Parse: parse(sql)
    Parse-->>Eng: AST
    
    Eng->>Plan: plan(AST, catalog)
    Plan->>Plan: Resolve table 't'
    Plan->>Plan: Bind expression 'x = 1'
    Plan->>Plan: Select access path (index or scan?)
    Plan-->>Eng: Plan (operator tree)
    
    Eng->>Exec: execute(plan)
    Exec->>Exec: Build RowCursor tree
    
    loop For each row
        Exec->>BTree: next_row()
        BTree->>Pager: read_page(page_id)
        Pager-->>BTree: &[u8] (borrowed page)
        BTree->>BTree: Decode row from page
        BTree-->>Exec: Row (registers)
        Exec->>Exec: Evaluate WHERE filter
        alt Passes filter
            Exec->>Exec: Collect row
        end
    end
    
    Exec-->>Eng: QueryResult
    Eng-->>App: Result with rows
```

## 4. Transaction Commit with Rollback Journal

This diagram illustrates the atomic commit protocol in rollback-journal mode.

```mermaid
sequenceDiagram
    participant App as Application
    participant Eng as Engine
    participant Pager as Pager
    participant Store as DiskStore
    participant Journal as -journal file
    participant DB as .db file

    App->>Eng: execute("BEGIN")
    Eng->>Pager: begin_transaction()
    Pager->>Pager: Create dirty page overlay
    
    App->>Eng: execute("INSERT INTO t VALUES (...)")
    Eng->>Pager: page_mut(page_id)
    Pager->>Pager: Add page to dirty overlay
    
    App->>Eng: execute("COMMIT")
    Eng->>Pager: commit()
    Pager->>Store: apply_commit(dirty_pages)
    
    Store->>Journal: Write pre-images of all dirty pages
    Store->>Journal: Fsync journal (+ directory if new)
    Store->>DB: Write modified pages in place
    Store->>DB: Fsync database
    Store->>Journal: Delete journal (commit point!)
    
    Store-->>Pager: Commit complete
    Pager-->>Eng: Success
    Eng-->>App: Ok(())
```

## 5. WAL Mode Commit and Checkpoint

This diagram shows how WAL mode handles commits and checkpoints.

```mermaid
sequenceDiagram
    participant App as Application
    participant Pager as Pager
    participant WalStore as WalStore
    participant WAL as -wal file
    participant DB as .db file

    Note over App,DB: COMMIT
    App->>Pager: commit()
    Pager->>WalStore: apply_commit(dirty_pages)
    
    loop For each dirty page
        WalStore->>WAL: Append frame (page_id, data)
    end
    
    WalStore->>WAL: Mark last frame as commit frame
    WalStore->>WAL: Fsync WAL (commit point!)
    WalStore-->>Pager: Commit complete
    
    Note over App,DB: Later: CHECKPOINT
    App->>Pager: checkpoint(mode=FULL)
    Pager->>WalStore: checkpoint(FULL)
    
    WalStore->>WalStore: Find frames up to oldest reader
    
    loop For each committed frame
        WalStore->>WAL: Read frame
        WalStore->>DB: Write page to .db file
    end
    
    WalStore->>DB: Fsync database
    WalStore->>WAL: Update WAL header (checkpoint point)
    WalStore->>WAL: Reset WAL to start
    WalStore-->>Pager: Checkpoint complete
```

## 6. B-tree Insert with Split

This diagram shows how B-tree insert handles page overflow.

```mermaid
graph TD
    Start([Insert new row]) --> TryInPlace{Cell fits in page?}
    
    TryInPlace -->|Yes| Splice["Splice cell in place<br/>(O(cell) edit)"]
    Splice --> Done([Done])
    
    TryInPlace -->|No| Rebuild["Rebuild page with new cell"]
    Rebuild --> CheckFull{Page still fits?}
    
    CheckFull -->|Yes| WritePage["Write modified page"]
    WritePage --> Done
    
    CheckFull -->|No| Split["N-way split:<br/>Divide cells among new pages"]
    Split --> PropagateUp["Insert separator into parent"]
    PropagateUp --> ParentFull{Parent fits?}
    
    ParentFull -->|Yes| UpdateParent["Update parent page"]
    UpdateParent --> Done
    
    ParentFull -->|No| RecursiveSplit["Recursively split parent"]
    RecursiveSplit --> AtRoot{At root?}
    
    AtRoot -->|No| PropagateUp
    AtRoot -->|Yes| GrowTree["Create new root page<br/>(tree grows one level)"]
    GrowTree --> Done
```

## 7. Foreign Key Cascade Flow

This diagram illustrates how foreign key CASCADE actions work.

```mermaid
graph TD
    Start([DELETE FROM parent WHERE id = X]) --> ParsePlan["Parse and plan DELETE"]
    ParsePlan --> CheckFK["Check catalog for FKs<br/>referencing parent"]
    
    CheckFK --> HasFK{FK with CASCADE?}
    HasFK -->|No| DirectDelete["Delete parent rows"]
    HasFK -->|Yes| CompileChild["Compile child DELETE program"]
    
    CompileChild --> ScanParent["Scan for parent rows to delete"]
    ScanParent --> ForEachRow["For each parent row"]
    
    ForEachRow --> ExecuteChild["Execute child DELETE<br/>(WHERE child.fk = parent.id)"]
    ExecuteChild --> ChildHasFK{Child has FKs?}
    
    ChildHasFK -->|Yes| RecursiveCascade["Recursively cascade to grandchildren"]
    ChildHasFK -->|No| DeleteChild["Delete child rows"]
    
    RecursiveCascade --> DeleteChild
    DeleteChild --> CheckBound{Recursion bound exceeded?}
    
    CheckBound -->|Yes| Error([Error: recursion limit])
    CheckBound -->|No| NextRow{More parent rows?}
    
    NextRow -->|Yes| ForEachRow
    NextRow -->|No| DeleteParent["Delete original parent rows"]
    
    DeleteParent --> UpdateIndexes["Update all affected indexes"]
    UpdateIndexes --> Done([Commit transaction])
    DirectDelete --> Done
```

## 8. Copy-on-Write Transaction Layer

This diagram shows how the COW layer manages dirty pages and savepoints.

```mermaid
graph TD
    Start([Transaction begins]) --> EmptyOverlay["Create empty dirty page overlay<br/>(HashMap)"]
    
    EmptyOverlay --> Read{Read page?}
    Read -->|Yes| InOverlay{Page in overlay?}
    InOverlay -->|Yes| ReturnDirty["Return &[u8] from overlay"]
    InOverlay -->|No| ReadStore["Read from committed store"]
    ReadStore --> ReturnClean["Return &[u8] from cache"]
    
    ReturnDirty --> NextOp
    ReturnClean --> NextOp[Next operation]
    
    NextOp --> Write{Write page?}
    Write -->|Yes| AlreadyDirty{Already in overlay?}
    AlreadyDirty -->|Yes| InPlaceEdit["Modify in place<br/>(O(edit) fast path)"]
    AlreadyDirty -->|No| CopyOnWrite["Clone page into overlay"]
    
    CopyOnWrite --> InPlaceEdit
    InPlaceEdit --> NextOp
    
    NextOp --> Savepoint{Create savepoint?}
    Savepoint -->|Yes| CaptureState["Capture pre-image delta"]
    CaptureState --> NextOp
    
    NextOp --> RollbackTo{Rollback to savepoint?}
    RollbackTo -->|Yes| RestoreDelta["Restore from pre-image delta"]
    RestoreDelta --> NextOp
    
    NextOp --> Commit{Commit?}
    Commit -->|Yes| HandToStore["Hand dirty pages to store.apply_commit()"]
    HandToStore --> ClearOverlay["Clear overlay"]
    
    Commit -->|No, Rollback| DropOverlay["Drop overlay"]
    
    ClearOverlay --> Done([Transaction complete])
    DropOverlay --> Done
```

All diagrams render correctly in GitHub-flavored Markdown.
