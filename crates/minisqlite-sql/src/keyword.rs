//! SQLite keyword table. The 147 keywords from `spec/sqlite-doc/lang_keywords.html`.
//!
//! Keywords are matched case-insensitively. The tokenizer tags a keyword token with
//! its `Keyword` id; the parser decides whether a given keyword may also stand in as
//! an identifier (SQLite lets many *non-reserved* keywords be used as names — the
//! `%fallback` behaviour of its grammar). `can_be_identifier` approximates that set:
//! it returns `true` for the clearly non-reserved keywords and `false` for the
//! structural / operator keywords that must stay reserved so core SQL never
//! misparses. This is a deliberate approximation of SQLite's exact fallback list;
//! it errs toward requiring a quote (a loud error) rather than silently misparsing.

/// Longest keyword is `CURRENT_TIMESTAMP` (17 bytes); anything longer is not a keyword.
const MAX_KEYWORD_LEN: usize = 17;

/// One SQLite keyword. Canonical spelling is uppercase (see [`Keyword::as_str`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    Abort,
    Action,
    Add,
    After,
    All,
    Alter,
    Always,
    Analyze,
    And,
    As,
    Asc,
    Attach,
    Autoincrement,
    Before,
    Begin,
    Between,
    By,
    Cascade,
    Case,
    Cast,
    Check,
    Collate,
    Column,
    Commit,
    Conflict,
    Constraint,
    Create,
    Cross,
    Current,
    CurrentDate,
    CurrentTime,
    CurrentTimestamp,
    Database,
    Default,
    Deferrable,
    Deferred,
    Delete,
    Desc,
    Detach,
    Distinct,
    Do,
    Drop,
    Each,
    Else,
    End,
    Escape,
    Except,
    Exclude,
    Exclusive,
    Exists,
    Explain,
    Fail,
    Filter,
    First,
    Following,
    For,
    Foreign,
    From,
    Full,
    Generated,
    Glob,
    Group,
    Groups,
    Having,
    If,
    Ignore,
    Immediate,
    In,
    Index,
    Indexed,
    Initially,
    Inner,
    Insert,
    Instead,
    Intersect,
    Into,
    Is,
    Isnull,
    Join,
    Key,
    Last,
    Left,
    Like,
    Limit,
    Match,
    Materialized,
    Natural,
    No,
    Not,
    Nothing,
    Notnull,
    Null,
    Nulls,
    Of,
    Offset,
    On,
    Or,
    Order,
    Others,
    Outer,
    Over,
    Partition,
    Plan,
    Pragma,
    Preceding,
    Primary,
    Query,
    Raise,
    Range,
    Recursive,
    References,
    Regexp,
    Reindex,
    Release,
    Rename,
    Replace,
    Restrict,
    Returning,
    Right,
    Rollback,
    Row,
    Rows,
    Savepoint,
    Select,
    Set,
    Table,
    Temp,
    Temporary,
    Then,
    Ties,
    To,
    Transaction,
    Trigger,
    Unbounded,
    Union,
    Unique,
    Update,
    Using,
    Vacuum,
    Values,
    View,
    Virtual,
    When,
    Where,
    Window,
    With,
    Without,
}

impl Keyword {
    /// Canonical uppercase spelling of this keyword.
    pub fn as_str(self) -> &'static str {
        use Keyword::*;
        match self {
            Abort => "ABORT",
            Action => "ACTION",
            Add => "ADD",
            After => "AFTER",
            All => "ALL",
            Alter => "ALTER",
            Always => "ALWAYS",
            Analyze => "ANALYZE",
            And => "AND",
            As => "AS",
            Asc => "ASC",
            Attach => "ATTACH",
            Autoincrement => "AUTOINCREMENT",
            Before => "BEFORE",
            Begin => "BEGIN",
            Between => "BETWEEN",
            By => "BY",
            Cascade => "CASCADE",
            Case => "CASE",
            Cast => "CAST",
            Check => "CHECK",
            Collate => "COLLATE",
            Column => "COLUMN",
            Commit => "COMMIT",
            Conflict => "CONFLICT",
            Constraint => "CONSTRAINT",
            Create => "CREATE",
            Cross => "CROSS",
            Current => "CURRENT",
            CurrentDate => "CURRENT_DATE",
            CurrentTime => "CURRENT_TIME",
            CurrentTimestamp => "CURRENT_TIMESTAMP",
            Database => "DATABASE",
            Default => "DEFAULT",
            Deferrable => "DEFERRABLE",
            Deferred => "DEFERRED",
            Delete => "DELETE",
            Desc => "DESC",
            Detach => "DETACH",
            Distinct => "DISTINCT",
            Do => "DO",
            Drop => "DROP",
            Each => "EACH",
            Else => "ELSE",
            End => "END",
            Escape => "ESCAPE",
            Except => "EXCEPT",
            Exclude => "EXCLUDE",
            Exclusive => "EXCLUSIVE",
            Exists => "EXISTS",
            Explain => "EXPLAIN",
            Fail => "FAIL",
            Filter => "FILTER",
            First => "FIRST",
            Following => "FOLLOWING",
            For => "FOR",
            Foreign => "FOREIGN",
            From => "FROM",
            Full => "FULL",
            Generated => "GENERATED",
            Glob => "GLOB",
            Group => "GROUP",
            Groups => "GROUPS",
            Having => "HAVING",
            If => "IF",
            Ignore => "IGNORE",
            Immediate => "IMMEDIATE",
            In => "IN",
            Index => "INDEX",
            Indexed => "INDEXED",
            Initially => "INITIALLY",
            Inner => "INNER",
            Insert => "INSERT",
            Instead => "INSTEAD",
            Intersect => "INTERSECT",
            Into => "INTO",
            Is => "IS",
            Isnull => "ISNULL",
            Join => "JOIN",
            Key => "KEY",
            Last => "LAST",
            Left => "LEFT",
            Like => "LIKE",
            Limit => "LIMIT",
            Match => "MATCH",
            Materialized => "MATERIALIZED",
            Natural => "NATURAL",
            No => "NO",
            Not => "NOT",
            Nothing => "NOTHING",
            Notnull => "NOTNULL",
            Null => "NULL",
            Nulls => "NULLS",
            Of => "OF",
            Offset => "OFFSET",
            On => "ON",
            Or => "OR",
            Order => "ORDER",
            Others => "OTHERS",
            Outer => "OUTER",
            Over => "OVER",
            Partition => "PARTITION",
            Plan => "PLAN",
            Pragma => "PRAGMA",
            Preceding => "PRECEDING",
            Primary => "PRIMARY",
            Query => "QUERY",
            Raise => "RAISE",
            Range => "RANGE",
            Recursive => "RECURSIVE",
            References => "REFERENCES",
            Regexp => "REGEXP",
            Reindex => "REINDEX",
            Release => "RELEASE",
            Rename => "RENAME",
            Replace => "REPLACE",
            Restrict => "RESTRICT",
            Returning => "RETURNING",
            Right => "RIGHT",
            Rollback => "ROLLBACK",
            Row => "ROW",
            Rows => "ROWS",
            Savepoint => "SAVEPOINT",
            Select => "SELECT",
            Set => "SET",
            Table => "TABLE",
            Temp => "TEMP",
            Temporary => "TEMPORARY",
            Then => "THEN",
            Ties => "TIES",
            To => "TO",
            Transaction => "TRANSACTION",
            Trigger => "TRIGGER",
            Unbounded => "UNBOUNDED",
            Union => "UNION",
            Unique => "UNIQUE",
            Update => "UPDATE",
            Using => "USING",
            Vacuum => "VACUUM",
            Values => "VALUES",
            View => "VIEW",
            Virtual => "VIRTUAL",
            When => "WHEN",
            Where => "WHERE",
            Window => "WINDOW",
            With => "WITH",
            Without => "WITHOUT",
        }
    }

    /// Look up a keyword by (case-insensitive) spelling. Returns `None` if the text
    /// is not a keyword (i.e. an ordinary identifier). ASCII-only: unquoted SQL
    /// identifiers are ASCII, so a stack uppercase buffer is enough and allocates
    /// nothing on the hot tokenizer path.
    pub fn lookup(s: &str) -> Option<Keyword> {
        let bytes = s.as_bytes();
        if bytes.is_empty() || bytes.len() > MAX_KEYWORD_LEN {
            return None;
        }
        let mut buf = [0u8; MAX_KEYWORD_LEN];
        for (i, &b) in bytes.iter().enumerate() {
            // A non-ASCII byte can never be part of a keyword.
            if !b.is_ascii() {
                return None;
            }
            buf[i] = b.to_ascii_uppercase();
        }
        // SAFETY-free: ascii-uppercased ascii bytes are valid UTF-8.
        let up = core::str::from_utf8(&buf[..bytes.len()]).ok()?;
        Self::from_upper(up)
    }

    fn from_upper(up: &str) -> Option<Keyword> {
        use Keyword::*;
        Some(match up {
            "ABORT" => Abort,
            "ACTION" => Action,
            "ADD" => Add,
            "AFTER" => After,
            "ALL" => All,
            "ALTER" => Alter,
            "ALWAYS" => Always,
            "ANALYZE" => Analyze,
            "AND" => And,
            "AS" => As,
            "ASC" => Asc,
            "ATTACH" => Attach,
            "AUTOINCREMENT" => Autoincrement,
            "BEFORE" => Before,
            "BEGIN" => Begin,
            "BETWEEN" => Between,
            "BY" => By,
            "CASCADE" => Cascade,
            "CASE" => Case,
            "CAST" => Cast,
            "CHECK" => Check,
            "COLLATE" => Collate,
            "COLUMN" => Column,
            "COMMIT" => Commit,
            "CONFLICT" => Conflict,
            "CONSTRAINT" => Constraint,
            "CREATE" => Create,
            "CROSS" => Cross,
            "CURRENT" => Current,
            "CURRENT_DATE" => CurrentDate,
            "CURRENT_TIME" => CurrentTime,
            "CURRENT_TIMESTAMP" => CurrentTimestamp,
            "DATABASE" => Database,
            "DEFAULT" => Default,
            "DEFERRABLE" => Deferrable,
            "DEFERRED" => Deferred,
            "DELETE" => Delete,
            "DESC" => Desc,
            "DETACH" => Detach,
            "DISTINCT" => Distinct,
            "DO" => Do,
            "DROP" => Drop,
            "EACH" => Each,
            "ELSE" => Else,
            "END" => End,
            "ESCAPE" => Escape,
            "EXCEPT" => Except,
            "EXCLUDE" => Exclude,
            "EXCLUSIVE" => Exclusive,
            "EXISTS" => Exists,
            "EXPLAIN" => Explain,
            "FAIL" => Fail,
            "FILTER" => Filter,
            "FIRST" => First,
            "FOLLOWING" => Following,
            "FOR" => For,
            "FOREIGN" => Foreign,
            "FROM" => From,
            "FULL" => Full,
            "GENERATED" => Generated,
            "GLOB" => Glob,
            "GROUP" => Group,
            "GROUPS" => Groups,
            "HAVING" => Having,
            "IF" => If,
            "IGNORE" => Ignore,
            "IMMEDIATE" => Immediate,
            "IN" => In,
            "INDEX" => Index,
            "INDEXED" => Indexed,
            "INITIALLY" => Initially,
            "INNER" => Inner,
            "INSERT" => Insert,
            "INSTEAD" => Instead,
            "INTERSECT" => Intersect,
            "INTO" => Into,
            "IS" => Is,
            "ISNULL" => Isnull,
            "JOIN" => Join,
            "KEY" => Key,
            "LAST" => Last,
            "LEFT" => Left,
            "LIKE" => Like,
            "LIMIT" => Limit,
            "MATCH" => Match,
            "MATERIALIZED" => Materialized,
            "NATURAL" => Natural,
            "NO" => No,
            "NOT" => Not,
            "NOTHING" => Nothing,
            "NOTNULL" => Notnull,
            "NULL" => Null,
            "NULLS" => Nulls,
            "OF" => Of,
            "OFFSET" => Offset,
            "ON" => On,
            "OR" => Or,
            "ORDER" => Order,
            "OTHERS" => Others,
            "OUTER" => Outer,
            "OVER" => Over,
            "PARTITION" => Partition,
            "PLAN" => Plan,
            "PRAGMA" => Pragma,
            "PRECEDING" => Preceding,
            "PRIMARY" => Primary,
            "QUERY" => Query,
            "RAISE" => Raise,
            "RANGE" => Range,
            "RECURSIVE" => Recursive,
            "REFERENCES" => References,
            "REGEXP" => Regexp,
            "REINDEX" => Reindex,
            "RELEASE" => Release,
            "RENAME" => Rename,
            "REPLACE" => Replace,
            "RESTRICT" => Restrict,
            "RETURNING" => Returning,
            "RIGHT" => Right,
            "ROLLBACK" => Rollback,
            "ROW" => Row,
            "ROWS" => Rows,
            "SAVEPOINT" => Savepoint,
            "SELECT" => Select,
            "SET" => Set,
            "TABLE" => Table,
            "TEMP" => Temp,
            "TEMPORARY" => Temporary,
            "THEN" => Then,
            "TIES" => Ties,
            "TO" => To,
            "TRANSACTION" => Transaction,
            "TRIGGER" => Trigger,
            "UNBOUNDED" => Unbounded,
            "UNION" => Union,
            "UNIQUE" => Unique,
            "UPDATE" => Update,
            "USING" => Using,
            "VACUUM" => Vacuum,
            "VALUES" => Values,
            "VIEW" => View,
            "VIRTUAL" => Virtual,
            "WHEN" => When,
            "WHERE" => Where,
            "WINDOW" => Window,
            "WITH" => With,
            "WITHOUT" => Without,
            _ => return None,
        })
    }

    /// Whether this keyword may be used as a bare (unquoted) identifier.
    ///
    /// SQLite lets most keywords fall back to identifiers; only a structural /
    /// operator core stays reserved. Listing the *reserved* set (and defaulting
    /// everything else to usable) matches SQLite's bias toward permissiveness while
    /// keeping the core statements unambiguous. Approximation — see module docs.
    pub fn can_be_identifier(self) -> bool {
        !self.is_reserved()
    }

    /// Reserved keywords: those that cannot serve as a bare identifier. Kept narrow
    /// and centred on the tokens that would otherwise be swallowed inside the core
    /// grammar (clause introducers, operators). Everything not listed here is treated
    /// as usable-as-identifier.
    ///
    /// The join words (`LEFT`/`RIGHT`/`FULL`/`INNER`/`OUTER`/`CROSS`/`NATURAL`) and the
    /// window words (`OVER`/`FILTER`/`WINDOW`) are DELIBERATELY not here: real SQLite
    /// permits all of them as ordinary identifiers (its tokenizer reclassifies `JOIN_KW`,
    /// `OVER`, and `WINDOW` to `TK_ID` in identifier positions, and `FILTER` via
    /// look-ahead — see `tokenize.c`). The grammar keeps them unambiguous positionally
    /// instead: join detection reads the raw keyword token, the window words are consumed
    /// straight after a `)` by the function-call parser, and a bare (no-`AS`) alias
    /// refuses the join words / a `WINDOW <name> AS` clause (see `parse_opt_alias`).
    pub fn is_reserved(self) -> bool {
        use Keyword::*;
        matches!(
            self,
            Add | All
                | Alter
                | And
                | As
                | Autoincrement
                | Between
                | Case
                | Check
                | Collate
                | Commit
                | Constraint
                | Create
                | Default
                | Deferrable
                | Delete
                | Distinct
                | Drop
                | Else
                | Escape
                | Except
                | Exists
                | Foreign
                | From
                | Group
                | Having
                | In
                | Index
                | Insert
                | Intersect
                | Into
                | Is
                | Isnull
                | Join
                | Limit
                | Not
                | Notnull
                | Null
                | On
                | Or
                | Order
                | Primary
                | References
                | Returning
                | Select
                | Set
                | Table
                | Then
                | To
                | Transaction
                | Union
                | Unique
                | Update
                | Using
                | Values
                | When
                | Where
                | With
        )
    }

    /// Whether this is a join operator keyword (`JOIN_KW` in SQLite's tokenizer):
    /// `NATURAL`/`LEFT`/`RIGHT`/`FULL`/`INNER`/`OUTER`/`CROSS`. These are legal
    /// identifiers (table/column names, `AS` aliases) but never a *bare* (no-`AS`)
    /// alias — in that position they always introduce a join, matching SQLite's
    /// `as ::= ids` rule which excludes `JOIN_KW`.
    pub fn is_join_keyword(self) -> bool {
        use Keyword::*;
        matches!(self, Natural | Left | Right | Full | Inner | Outer | Cross)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(Keyword::lookup("select"), Some(Keyword::Select));
        assert_eq!(Keyword::lookup("SELECT"), Some(Keyword::Select));
        assert_eq!(Keyword::lookup("SeLeCt"), Some(Keyword::Select));
        assert_eq!(Keyword::lookup("current_timestamp"), Some(Keyword::CurrentTimestamp));
        assert_eq!(Keyword::lookup("notakeyword"), None);
        assert_eq!(Keyword::lookup(""), None);
        // Too long to be any keyword.
        assert_eq!(Keyword::lookup("current_timestamp_x"), None);
    }

    #[test]
    fn as_str_round_trips_through_lookup() {
        // Every keyword's canonical spelling must look up back to itself.
        let all = [
            Keyword::Abort,
            Keyword::CurrentTimestamp,
            Keyword::Without,
            Keyword::Select,
            Keyword::Values,
        ];
        for kw in all {
            assert_eq!(Keyword::lookup(kw.as_str()), Some(kw), "{}", kw.as_str());
        }
    }

    #[test]
    fn reserved_vs_identifier() {
        assert!(Keyword::Select.is_reserved());
        assert!(Keyword::From.is_reserved());
        assert!(!Keyword::Key.is_reserved());
        assert!(Keyword::Key.can_be_identifier());
        assert!(!Keyword::Where.can_be_identifier());
    }

    #[test]
    fn join_and_window_keywords_are_identifiers() {
        // SQLite reclassifies JOIN_KW / OVER / WINDOW to identifiers in name positions,
        // and FILTER via look-ahead; none is reserved. (Only their grammatical position
        // makes them a join/window operator — see the parser.)
        for kw in [
            Keyword::Left,
            Keyword::Right,
            Keyword::Full,
            Keyword::Inner,
            Keyword::Outer,
            Keyword::Cross,
            Keyword::Natural,
            Keyword::Over,
            Keyword::Filter,
            Keyword::Window,
        ] {
            assert!(kw.can_be_identifier(), "{} must be usable as an identifier", kw.as_str());
        }
        // The seven join words are JOIN_KW (never a bare alias); OVER/FILTER/WINDOW are not.
        for kw in [
            Keyword::Left,
            Keyword::Right,
            Keyword::Full,
            Keyword::Inner,
            Keyword::Outer,
            Keyword::Cross,
            Keyword::Natural,
        ] {
            assert!(kw.is_join_keyword(), "{} is a join keyword", kw.as_str());
        }
        assert!(!Keyword::Over.is_join_keyword());
        assert!(!Keyword::Window.is_join_keyword());
        assert!(!Keyword::Join.is_join_keyword(), "JOIN itself stays reserved");
        assert!(Keyword::Join.is_reserved());
    }

    #[test]
    fn fallback_keywords_are_identifiers() {
        // EACH / GENERATED / ALWAYS are in SQLite's `%fallback ID` list (parse.y), so
        // they are usable as bare identifiers — only positional (FOR EACH ROW, GENERATED
        // ALWAYS AS) does the parser read them as keywords.
        assert!(Keyword::Each.can_be_identifier());
        assert!(Keyword::Generated.can_be_identifier());
        assert!(Keyword::Always.can_be_identifier());
    }
}
