//! Surgical text rewrites of stored `CREATE TABLE` / `CREATE INDEX` `sql` for
//! `ALTER TABLE`. These are the pure functional core of the ALTER path: each takes
//! SQL text plus a name/definition and returns the edited text, with no catalog or
//! pager state, so they are exhaustively unit-testable in isolation.
//!
//! Why token spans, not an AST re-render: SQLite records the *verbatim* `CREATE`
//! text in `sqlite_schema.sql`, and re-rendering it from a parsed AST would lose the
//! user's exact spelling, whitespace, and comments. Instead each function tokenizes
//! the text (tokens carry byte spans into the source, `token.start..token.end`),
//! locates the ONE token to change, and splices the original string around it — so
//! everything else is preserved byte-for-byte, exactly as ALTER TABLE does in
//! SQLite (`lang_altertable.html`: "works by modifying the SQL text of the schema").

use minisqlite_sql::{tokenize, Keyword, Token, TokenKind};
use minisqlite_types::{Error, Result};

/// Extract the verbatim column-definition text from an `ALTER TABLE ... ADD [COLUMN]
/// <column-def>` statement. Tokenize, find the `ADD` keyword, skip an optional
/// `COLUMN` keyword, then slice from the next token's start through the last real
/// token's end (excluding a trailing `;`/EOF). The returned slice is the exact
/// bytes to splice into the target table's stored column list.
pub(crate) fn extract_add_column_def(alter_sql: &str) -> Result<String> {
    let toks = tokenize(alter_sql)?;
    let add_idx = toks
        .iter()
        .position(|t| t.keyword() == Some(Keyword::Add))
        .ok_or_else(|| Error::Sql("ALTER TABLE ADD COLUMN: missing ADD keyword".into()))?;

    // The column definition begins after ADD (and an optional COLUMN keyword).
    let mut start_idx = add_idx + 1;
    if toks.get(start_idx).and_then(Token::keyword) == Some(Keyword::Column) {
        start_idx += 1;
    }
    let first = toks.get(start_idx).filter(|t| !is_terminator(&t.kind)).ok_or_else(|| {
        Error::Sql("ALTER TABLE ADD COLUMN: missing column definition".into())
    })?;
    let start = first.start;

    // The definition runs to the last token before a terminating `;` / EOF.
    let mut end = first.end;
    for t in &toks[start_idx..] {
        if is_terminator(&t.kind) {
            break;
        }
        end = t.end;
    }
    Ok(alter_sql[start..end].to_string())
}

/// Splice a new column definition into a stored `CREATE TABLE`'s column list.
///
/// The new column is inserted at the END of the column-definition list: right
/// before the FIRST top-level table constraint, or — when the table has none —
/// right before the column list's closing `)`. This ordering is required, not
/// cosmetic. A `CREATE TABLE` body is "one or more column definitions, optionally
/// followed by a list of table constraints" (`lang_createtable.html` §3), so a
/// column definition placed AFTER a table constraint is a syntax error that real
/// SQLite rejects when it re-parses `sqlite_schema.sql` — even though this engine's
/// own lenient parser accepts the interleaving. Inserting before the closing `)`
/// unconditionally (the naive choice) therefore corrupts every table that carries a
/// trailing `PRIMARY KEY(...)`/`UNIQUE(...)`/`CHECK(...)`/`FOREIGN KEY(...)`/
/// `CONSTRAINT ...`, so we splice before that constraint tail instead.
///
/// The column list is delimited by the FIRST top-level `(` after the table name and
/// its matching `)`, found by tracking paren depth, so nested parens from
/// `CHECK(...)`, `DECIMAL(10,2)`, etc. never confuse the scan and trailing options
/// (`WITHOUT ROWID` / `STRICT`) after the `)` are preserved. A table constraint is
/// identified by the LEADING token of a depth-1 list item: `CONSTRAINT`, `PRIMARY`,
/// `UNIQUE`, `CHECK`, `FOREIGN` are all reserved keywords, so an unquoted column
/// name can never be one — and the leading-token test is what distinguishes a
/// *table* constraint from a *column-level* one (`a INTEGER PRIMARY KEY`, whose item
/// leads with the column name `a`, stays a column def and is left before the `)`).
pub(crate) fn splice_column_into_create(create_sql: &str, column_def: &str) -> Result<String> {
    let toks = tokenize(create_sql)?;
    let open_idx = toks
        .iter()
        .position(|t| t.kind == TokenKind::LParen)
        .ok_or_else(|| Error::Sql("CREATE TABLE has no column list to add a column to".into()))?;

    // Walk the column list tracking paren depth. `at_item_start` marks the first
    // token of a depth-1 list item (immediately after the opening `(` or a depth-1
    // `,`). Stop at the first item that leads with a table-constraint keyword
    // (splice before it) or, failing that, at the closing `)`.
    let mut depth: usize = 0;
    let mut at_item_start = false;
    // (byte offset to splice at, whether it is before a table constraint).
    let mut splice: Option<(usize, bool)> = None;
    for t in &toks[open_idx..] {
        match t.kind {
            TokenKind::LParen => {
                depth += 1;
                if depth == 1 {
                    at_item_start = true;
                }
                continue;
            }
            TokenKind::RParen => {
                depth -= 1;
                if depth == 0 {
                    splice = Some((t.start, false));
                    break;
                }
                continue;
            }
            TokenKind::Comma if depth == 1 => {
                at_item_start = true;
                continue;
            }
            _ => {}
        }
        if at_item_start && depth == 1 {
            at_item_start = false;
            if is_table_constraint_lead(t) {
                splice = Some((t.start, true));
                break;
            }
        }
    }
    let (pos, before_constraint) = splice
        .ok_or_else(|| Error::Sql("CREATE TABLE column list has no matching )".into()))?;

    // Both shapes leave exactly one ", " between the new column and its neighbours;
    // only which side carries the separator differs:
    //   before a constraint: "..., <col>, CHECK(...)"  (col + ", " before the item)
    //   before the ")":       "..., <col>)"            (", " + col before the paren)
    let mut out = String::with_capacity(create_sql.len() + column_def.len() + 2);
    out.push_str(&create_sql[..pos]);
    if before_constraint {
        out.push_str(column_def);
        out.push_str(", ");
    } else {
        out.push_str(", ");
        out.push_str(column_def);
    }
    out.push_str(&create_sql[pos..]);
    Ok(out)
}

/// Does this token begin a table constraint? A table-constraint list item leads
/// with `CONSTRAINT name?` then `PRIMARY KEY`/`UNIQUE`/`CHECK`/`FOREIGN KEY`; all
/// five leading keywords are reserved, so an unquoted column name can never collide
/// with them and the leading token alone tells a table constraint from a column
/// def. Meaningful ONLY for the first token of a depth-1 list item — the same
/// keyword appearing mid-item (a column-level `... PRIMARY KEY`) is not a table
/// constraint and must not be treated as one.
fn is_table_constraint_lead(tok: &Token) -> bool {
    matches!(
        tok.keyword(),
        Some(
            Keyword::Constraint
                | Keyword::Primary
                | Keyword::Unique
                | Keyword::Check
                | Keyword::Foreign
        )
    )
}

/// Rewrite the table name in a stored `CREATE TABLE` to `new_name`.
///
/// The name token is the first identifier after the `TABLE` keyword, skipping an
/// optional `IF NOT EXISTS`; a `schema.name` qualifier is handled by taking the token
/// after the `.`. The token's byte span is replaced with `new_name`, quoted only if
/// necessary. Everything else (columns, constraints, options, whitespace) is
/// preserved verbatim.
pub(crate) fn rewrite_create_table_name(create_sql: &str, new_name: &str) -> Result<String> {
    let toks = tokenize(create_sql)?;
    let table_idx = toks
        .iter()
        .position(|t| t.keyword() == Some(Keyword::Table))
        .ok_or_else(|| Error::Sql("CREATE TABLE: missing TABLE keyword".into()))?;

    // Skip an optional `IF NOT EXISTS` between TABLE and the name.
    let mut idx = table_idx + 1;
    if toks.get(idx).and_then(Token::keyword) == Some(Keyword::If) {
        idx += 1;
        if toks.get(idx).and_then(Token::keyword) == Some(Keyword::Not) {
            idx += 1;
        }
        if toks.get(idx).and_then(Token::keyword) == Some(Keyword::Exists) {
            idx += 1;
        }
    }
    let name_idx = qualified_name_index(&toks, idx);
    replace_token_span(create_sql, &toks, name_idx, new_name, "CREATE TABLE table name")
}

/// Rewrite the target-table name in a stored `CREATE INDEX ... ON <table> (...)` to
/// `new_name`. The table name is the first identifier after the `ON` keyword
/// (handling a `schema.name` qualifier); its byte span is replaced, quoted as
/// needed, and the rest of the index definition is preserved verbatim.
pub(crate) fn rewrite_index_table_name(index_sql: &str, new_name: &str) -> Result<String> {
    let toks = tokenize(index_sql)?;
    let on_idx = toks
        .iter()
        .position(|t| t.keyword() == Some(Keyword::On))
        .ok_or_else(|| Error::Sql("CREATE INDEX: missing ON keyword".into()))?;
    let name_idx = qualified_name_index(&toks, on_idx + 1);
    replace_token_span(index_sql, &toks, name_idx, new_name, "CREATE INDEX target table")
}

/// Rewrite the target-table name in a stored `CREATE TRIGGER ... ON <table> ...` to
/// `new_name`, for `ALTER TABLE ... RENAME TO`. Same shape as
/// [`rewrite_index_table_name`]: the target table is the first identifier after the
/// FIRST `ON` keyword (handling a `schema.name` qualifier); its byte span is
/// replaced, quoted as needed, and everything else — the trigger's own name, the
/// `BEFORE`/`AFTER`/`INSTEAD OF`, the event, its `WHEN` and `BEGIN...END` body — is
/// preserved verbatim.
///
/// The FIRST `Keyword::On` is always the target table, never a later `ON` in the
/// body: the `CREATE TRIGGER` grammar before the target is
/// `CREATE [TEMP] TRIGGER [IF NOT EXISTS] [schema.]name
/// [BEFORE|AFTER|INSTEAD OF] {DELETE|INSERT|UPDATE [OF col,...]} ON [schema.]table`.
/// `INSTEAD OF` and `UPDATE OF col` use the keyword `OF`, not `ON`, and a join `ON`
/// inside the trigger body comes strictly AFTER `ON <table>` — so no `ON` precedes
/// the target.
///
/// This rewrites ONLY the target table (the `ON <table>` and, via the caller, the
/// row's `tbl_name`), which is all `load_trigger_row` validates on reload. References
/// to the renamed table INSIDE the trigger body / `WHEN` are a deliberate,
/// load-tolerant deferral (see `alter_rename_to`).
pub(crate) fn rewrite_trigger_table_name(trigger_sql: &str, new_name: &str) -> Result<String> {
    let toks = tokenize(trigger_sql)?;
    let on_idx = toks
        .iter()
        .position(|t| t.keyword() == Some(Keyword::On))
        .ok_or_else(|| Error::Sql("CREATE TRIGGER: missing ON keyword".into()))?;
    let name_idx = qualified_name_index(&toks, on_idx + 1);
    replace_token_span(trigger_sql, &toks, name_idx, new_name, "CREATE TRIGGER target table")
}

/// Rename every reference to column `from` (ASCII case-insensitive) to `to` inside a
/// stored `CREATE TABLE`, for `ALTER TABLE ... RENAME COLUMN`. Rewrites the column's
/// own definition name AND every reference to it within a table constraint
/// (`PRIMARY KEY(from)`, `UNIQUE(from)`, `CHECK(... from ...)`,
/// `FOREIGN KEY(from)`) or a generated-column expression — i.e. every identifier
/// token naming the column within the column-list/constraint region — quoting `to`
/// as needed so the result re-parses.
///
/// The region is the first top-level `(...)` after the table name (the column list
/// plus table constraints); the table's own name, before that `(`, is never touched.
/// Only bare and quoted *identifier* tokens are matched (`TokenKind::Ident` /
/// `QuotedIdent`); a keyword token is never rewritten, so the `KEY` of `PRIMARY KEY`
/// or a type keyword can never be clobbered by a column that happens to be named the
/// same, and a string literal (`TokenKind::Str`) is not an identifier so it is safe.
///
/// Bounded-approach limits vs SQLite's fully scope-aware rename (documented, not a
/// silent gap): a declared TYPE literally named the same as the column
/// (`CREATE TABLE t(b b)`), or a `REFERENCES other(from)` target column of ANOTHER
/// table that happens to share the name, would also be rewritten here. Both are
/// pathological and outside the conformance cases; the faithful version needs the
/// binder to resolve each identifier's scope.
pub(crate) fn rename_column_in_create(create_sql: &str, from: &str, to: &str) -> Result<String> {
    let toks = tokenize(create_sql)?;
    let open_idx = toks
        .iter()
        .position(|t| t.kind == TokenKind::LParen)
        .ok_or_else(|| Error::Sql("CREATE TABLE has no column list".into()))?;
    let close_idx = matching_close_paren(&toks, open_idx)?;
    // The region is strictly between the outer `(` and its matching `)`, so every
    // column def and table constraint (including nested `CHECK(...)` parens) is
    // covered while the table name and trailing options stay untouched.
    let region = &toks[open_idx + 1..close_idx];
    Ok(replace_matching_idents(create_sql, region, from, to))
}

/// Rename column `from` -> `to` (ASCII case-insensitive) inside a stored
/// `CREATE INDEX`'s indexed-column list AND its partial-index `WHERE` predicate, so
/// a dependent index follows a table's `RENAME COLUMN`. Everything up to and
/// including the target-table name (the `CREATE [UNIQUE] INDEX <name> ON <table>`
/// prefix) is left verbatim — only the region AFTER the table name, where the
/// index's columns and `WHERE` live, is rewritten. Same identifier-token matching
/// and bounded limits as [`rename_column_in_create`].
pub(crate) fn rename_column_in_index(index_sql: &str, from: &str, to: &str) -> Result<String> {
    let toks = tokenize(index_sql)?;
    let on_idx = toks
        .iter()
        .position(|t| t.keyword() == Some(Keyword::On))
        .ok_or_else(|| Error::Sql("CREATE INDEX: missing ON keyword".into()))?;
    let name_idx = qualified_name_index(&toks, on_idx + 1);
    // Everything after the target-table name: the `(col, ...)` list and any
    // `WHERE <expr>`. The trailing `Eof` token decodes to no name, so including it is
    // harmless.
    let region = toks.get(name_idx + 1..).unwrap_or(&[]);
    Ok(replace_matching_idents(index_sql, region, from, to))
}

/// Remove the definition of column `col` (ASCII case-insensitive) from a stored
/// `CREATE TABLE`, for `ALTER TABLE ... DROP COLUMN`. Deletes the column's depth-1
/// list item plus exactly ONE adjoining comma (the trailing comma when the column is
/// not the last list item, otherwise the leading comma), so the remaining columns,
/// table constraints, whitespace, and trailing options (`WITHOUT ROWID` / `STRICT`)
/// stay intact and the result re-parses with one fewer column.
///
/// Only a *column definition* is ever removed: a depth-1 list item is a column def iff
/// its LEADING token is an identifier (`Ident` / `QuotedIdent`); a table constraint
/// leads with a reserved keyword (`CONSTRAINT`/`PRIMARY`/`UNIQUE`/`CHECK`/`FOREIGN`),
/// so this never deletes a constraint that merely *references* `col`. (The caller in
/// `schemacatalog` rejects dropping a column named by any surviving constraint/index
/// before this runs, so a leftover reference to the dropped name cannot occur.)
///
/// Fails closed if the column is not found or is the table's only list item (the
/// latter is also rejected upstream with SQLite's wording; guarding here keeps the
/// function total and panic-free).
pub(crate) fn remove_column_from_create(create_sql: &str, col: &str) -> Result<String> {
    let toks = tokenize(create_sql)?;
    let open_idx = toks
        .iter()
        .position(|t| t.kind == TokenKind::LParen)
        .ok_or_else(|| Error::Sql("CREATE TABLE has no column list".into()))?;
    let close_idx = matching_close_paren(&toks, open_idx)?;
    let items = depth1_items(&toks, open_idx, close_idx);

    let target = items
        .iter()
        .position(|&(first, _)| {
            token_ident_name(&toks[first], create_sql)
                .is_some_and(|name| name.eq_ignore_ascii_case(col))
        })
        .ok_or_else(|| Error::Sql(format!("no such column: \"{col}\"")))?;

    if items.len() == 1 {
        return Err(Error::Sql(format!(
            "cannot drop column \"{col}\": no other columns exist"
        )));
    }

    let (first, last) = items[target];
    // Delete the item together with exactly one adjoining comma so the surviving list
    // keeps a single ", " between each neighbour: a non-last item takes the comma and
    // gap AFTER it (up to the next item's first token); the last item takes the comma
    // and gap BEFORE it (from the previous item's last token).
    let (del_start, del_end) = if target + 1 < items.len() {
        (toks[first].start, toks[items[target + 1].0].start)
    } else {
        (toks[items[target - 1].1].end, toks[last].end)
    };

    let mut out = String::with_capacity(create_sql.len());
    out.push_str(&create_sql[..del_start]);
    out.push_str(&create_sql[del_end..]);
    Ok(out)
}

/// The `(first, last)` token indices (inclusive) of each depth-1 list item between the
/// column-list `(` at `open_idx` and its matching `)` at `close_idx`. Items are
/// separated by depth-1 commas; a nested `(...)` (a `CHECK(...)`, `DECIMAL(10,2)`) is
/// part of whichever item encloses it, and a comma inside a string literal is a single
/// `Str` token so it never splits an item.
fn depth1_items(toks: &[Token], open_idx: usize, close_idx: usize) -> Vec<(usize, usize)> {
    let mut items = Vec::new();
    let mut depth: usize = 0;
    let mut item_start: Option<usize> = None;
    for i in open_idx..=close_idx {
        match toks[i].kind {
            TokenKind::LParen => {
                depth += 1;
                if depth == 1 {
                    // The outer `(` itself is not part of any item.
                    continue;
                }
            }
            TokenKind::RParen => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = item_start.take() {
                        items.push((s, i - 1));
                    }
                    break;
                }
            }
            TokenKind::Comma if depth == 1 => {
                if let Some(s) = item_start.take() {
                    items.push((s, i - 1));
                }
                continue;
            }
            _ => {}
        }
        if depth >= 1 && item_start.is_none() && toks[i].kind != TokenKind::LParen {
            item_start = Some(i);
        }
    }
    items
}

/// The index of the `)` that closes the `(` at `open_idx`, tracking paren depth so a
/// nested `(...)` (a `CHECK(...)`, `DECIMAL(10,2)`) does not end the scan early.
fn matching_close_paren(toks: &[Token], open_idx: usize) -> Result<usize> {
    let mut depth: usize = 0;
    for (i, t) in toks.iter().enumerate().skip(open_idx) {
        match t.kind {
            TokenKind::LParen => depth += 1,
            TokenKind::RParen => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }
    Err(Error::Sql("unbalanced parentheses in stored CREATE statement".into()))
}

/// The decoded identifier name a token spells, or `None` when it is not an
/// identifier. A bare `Ident` recovers its spelling from the source slice (case
/// preserved); a `QuotedIdent` carries its already-decoded value. A keyword token is
/// deliberately NOT an identifier here: matching it would let a column named like a
/// keyword rewrite the `KEY`/`PRIMARY`/type keywords around it.
fn token_ident_name(tok: &Token, sql: &str) -> Option<String> {
    match &tok.kind {
        TokenKind::Ident => Some(tok.text(sql).to_string()),
        TokenKind::QuotedIdent { value, .. } => Some(value.clone()),
        _ => None,
    }
}

/// Splice `sql`, replacing the byte span of every identifier token in `region` whose
/// decoded name equals `from` (ASCII case-insensitive) with `to`, quoted only if
/// needed. `region` is a contiguous token slice of `sql`; tokens outside it (and
/// non-matching tokens inside it) are copied verbatim, so all whitespace, comments,
/// and unrelated identifiers are preserved. The walk is left-to-right and each
/// replaced span is disjoint, so byte offsets stay valid as the output is built.
fn replace_matching_idents(sql: &str, region: &[Token], from: &str, to: &str) -> String {
    let replacement = quote_ident_if_needed(to);
    let mut out = String::with_capacity(sql.len());
    let mut cursor = 0usize;
    for t in region {
        if token_ident_name(t, sql).is_some_and(|name| name.eq_ignore_ascii_case(from)) {
            out.push_str(&sql[cursor..t.start]);
            out.push_str(&replacement);
            cursor = t.end;
        }
    }
    out.push_str(&sql[cursor..]);
    out
}

/// Given the index of the first token of a (possibly schema-qualified) object name,
/// return the index of the token that is the *object* name itself: if a `.` follows
/// the first token, the real name is the token after the `.` (skip the `schema`
/// qualifier), otherwise it is the first token.
fn qualified_name_index(toks: &[Token], first: usize) -> usize {
    if toks.get(first + 1).map(|t| &t.kind) == Some(&TokenKind::Dot) {
        first + 2
    } else {
        first
    }
}

/// Replace the byte span of the identifier token at `idx` with `new_name` (quoted as
/// needed), returning the edited SQL. Fails closed if the token is missing or is not
/// an identifier-like token — the caller located a name, so anything else means the
/// stored SQL did not have the shape we expect (corrupt/foreign), not a name to edit.
fn replace_token_span(
    sql: &str,
    toks: &[Token],
    idx: usize,
    new_name: &str,
    what: &str,
) -> Result<String> {
    let tok = toks
        .get(idx)
        .ok_or_else(|| Error::Sql(format!("could not locate the {what} to rename")))?;
    let is_name = matches!(
        tok.kind,
        TokenKind::Ident | TokenKind::QuotedIdent { .. } | TokenKind::Keyword(_)
    );
    if !is_name {
        return Err(Error::Sql(format!(
            "unexpected token where the {what} was expected: {:?}",
            tok.text(sql)
        )));
    }
    let replacement = quote_ident_if_needed(new_name);
    let mut out = String::with_capacity(sql.len() + replacement.len());
    out.push_str(&sql[..tok.start]);
    out.push_str(&replacement);
    out.push_str(&sql[tok.end..]);
    Ok(out)
}

/// Render `name` as a SQL identifier: bare when it is a plain identifier that is not
/// a keyword, else double-quoted with any embedded `"` doubled. Quoting a keyword or
/// a name with unusual characters is what keeps the rewritten `sql` re-parseable.
fn quote_ident_if_needed(name: &str) -> String {
    if is_plain_identifier(name) && Keyword::lookup(name).is_none() {
        return name.to_string();
    }
    let mut out = String::with_capacity(name.len() + 2);
    out.push('"');
    for ch in name.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
    out
}

/// A plain ASCII identifier: a leading letter/`_` then letters/digits/`_`. Anything
/// else (empty, leading digit, non-ASCII, punctuation, spaces) needs quoting.
fn is_plain_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// A statement-terminating token: the trailing `;` or the EOF sentinel.
fn is_terminator(kind: &TokenKind) -> bool {
    matches!(kind, TokenKind::Semicolon | TokenKind::Eof)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- extract_add_column_def ------------------------------------------------

    #[test]
    fn extract_add_column_with_and_without_column_keyword() {
        assert_eq!(
            extract_add_column_def("ALTER TABLE t ADD COLUMN c INTEGER").unwrap(),
            "c INTEGER"
        );
        assert_eq!(extract_add_column_def("ALTER TABLE t ADD c INTEGER").unwrap(), "c INTEGER");
    }

    #[test]
    fn extract_add_column_preserves_constraints_and_defaults_verbatim() {
        assert_eq!(
            extract_add_column_def("ALTER TABLE t ADD COLUMN c TEXT DEFAULT 'a b'").unwrap(),
            "c TEXT DEFAULT 'a b'"
        );
        // Nested parens inside the column def (a CHECK) are kept whole.
        assert_eq!(
            extract_add_column_def("ALTER TABLE t ADD c INT CHECK (c > 0)").unwrap(),
            "c INT CHECK (c > 0)"
        );
        // A quoted column name is preserved with its quotes.
        assert_eq!(
            extract_add_column_def("ALTER TABLE t ADD COLUMN \"my col\" TEXT").unwrap(),
            "\"my col\" TEXT"
        );
    }

    #[test]
    fn extract_add_column_ignores_trailing_semicolon() {
        assert_eq!(extract_add_column_def("ALTER TABLE t ADD c INT;").unwrap(), "c INT");
        assert_eq!(extract_add_column_def("ALTER TABLE t ADD c INT ;").unwrap(), "c INT");
    }

    #[test]
    fn extract_add_column_errors_when_no_definition() {
        assert!(extract_add_column_def("ALTER TABLE t ADD COLUMN").is_err());
        assert!(extract_add_column_def("ALTER TABLE t RENAME TO x").is_err());
    }

    // --- splice_column_into_create --------------------------------------------

    #[test]
    fn splice_appends_before_closing_paren() {
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a INTEGER)", "b TEXT").unwrap(),
            "CREATE TABLE t(a INTEGER, b TEXT)"
        );
    }

    #[test]
    fn splice_lands_before_the_outer_paren_with_nested_parens() {
        // DECIMAL(10,2): the inner `)` must not be mistaken for the column list's.
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a DECIMAL(10,2))", "b TEXT").unwrap(),
            "CREATE TABLE t(a DECIMAL(10,2), b TEXT)"
        );
    }

    #[test]
    fn splice_lands_before_the_first_table_constraint() {
        // A table constraint must stay LAST: real sqlite rejects a column def after
        // a table constraint, so the new column goes before it, not before the `)`.
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a INT, CHECK(a > 0))", "b TEXT").unwrap(),
            "CREATE TABLE t(a INT, b TEXT, CHECK(a > 0))"
        );
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a, b, PRIMARY KEY(a))", "c").unwrap(),
            "CREATE TABLE t(a, b, c, PRIMARY KEY(a))"
        );
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a, UNIQUE(a))", "b TEXT").unwrap(),
            "CREATE TABLE t(a, b TEXT, UNIQUE(a))"
        );
        assert_eq!(
            splice_column_into_create(
                "CREATE TABLE t(a, FOREIGN KEY(a) REFERENCES u(x))",
                "b"
            )
            .unwrap(),
            "CREATE TABLE t(a, b, FOREIGN KEY(a) REFERENCES u(x))"
        );
        // A named CONSTRAINT leads the tail too.
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a, CONSTRAINT pk PRIMARY KEY(a))", "b")
                .unwrap(),
            "CREATE TABLE t(a, b, CONSTRAINT pk PRIMARY KEY(a))"
        );
        // Only the FIRST constraint matters; everything from it on is the tail.
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a, UNIQUE(a), CHECK(a > 0))", "b").unwrap(),
            "CREATE TABLE t(a, b, UNIQUE(a), CHECK(a > 0))"
        );
    }

    #[test]
    fn splice_does_not_mistake_a_column_level_constraint_for_a_table_one() {
        // `PRIMARY KEY` / `CHECK` appearing MID-item (a column-level constraint) is
        // not a table constraint: the item leads with the column name, so the new
        // column is still appended before the `)`, producing valid sqlite.
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a INTEGER PRIMARY KEY)", "b").unwrap(),
            "CREATE TABLE t(a INTEGER PRIMARY KEY, b)"
        );
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a INT CHECK(a > 0))", "b").unwrap(),
            "CREATE TABLE t(a INT CHECK(a > 0), b)"
        );
    }

    #[test]
    fn spliced_sql_reparses_with_columns_before_constraints() {
        // Property: after splicing into a table WITH a table constraint, the result
        // re-parses and every column def precedes every table constraint (the exact
        // invariant real sqlite enforces on the stored schema).
        for (create, col) in [
            ("CREATE TABLE t(a, b, PRIMARY KEY(a, b))", "c INTEGER"),
            ("CREATE TABLE t(a INT, CHECK(a > 0))", "b TEXT DEFAULT 'x'"),
            ("CREATE TABLE t(a, CONSTRAINT u UNIQUE(a))", "b"),
        ] {
            let out = splice_column_into_create(create, col).unwrap();
            let ast = minisqlite_sql::parse(&out)
                .unwrap_or_else(|e| panic!("spliced sql {out:?} must re-parse: {e}"));
            let ct = match ast.statements.as_slice() {
                [minisqlite_sql::Statement::CreateTable(ct)] => ct,
                other => panic!("expected one CREATE TABLE, got {other:?}"),
            };
            let columns = match &ct.body {
                minisqlite_sql::CreateTableBody::Columns { columns, constraints, .. } => {
                    assert!(!constraints.is_empty(), "test input has a table constraint");
                    columns
                }
                other => panic!("expected a column-list body, got {other:?}"),
            };
            // The new column is present and is the LAST column def (so it precedes
            // every table constraint, which is what real sqlite requires).
            let last = columns.last().expect("at least one column");
            assert!(
                col.starts_with(&last.name),
                "new column {col:?} should be the final column def, got {:?}",
                last.name
            );
        }
    }

    #[test]
    fn splice_preserves_trailing_table_options() {
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a) WITHOUT ROWID", "b TEXT").unwrap(),
            "CREATE TABLE t(a, b TEXT) WITHOUT ROWID"
        );
        assert_eq!(
            splice_column_into_create("CREATE TABLE t(a) STRICT", "b INT").unwrap(),
            "CREATE TABLE t(a, b INT) STRICT"
        );
    }

    #[test]
    fn splice_handles_quoted_table_name_containing_a_paren() {
        // The `(` inside a quoted identifier is part of the token, not a real LParen,
        // so the first real `(` is still the column list opener.
        assert_eq!(
            splice_column_into_create("CREATE TABLE \"t(x\"(a)", "b").unwrap(),
            "CREATE TABLE \"t(x\"(a, b)"
        );
    }

    // --- rewrite_create_table_name --------------------------------------------

    #[test]
    fn rewrite_table_name_basic() {
        assert_eq!(
            rewrite_create_table_name("CREATE TABLE t(a INTEGER)", "t2").unwrap(),
            "CREATE TABLE t2(a INTEGER)"
        );
    }

    #[test]
    fn rewrite_table_name_skips_if_not_exists() {
        assert_eq!(
            rewrite_create_table_name("CREATE TABLE IF NOT EXISTS t (a)", "t2").unwrap(),
            "CREATE TABLE IF NOT EXISTS t2 (a)"
        );
    }

    #[test]
    fn rewrite_table_name_replaces_quoted_name() {
        assert_eq!(
            rewrite_create_table_name("CREATE TABLE \"old name\"(a)", "new").unwrap(),
            "CREATE TABLE new(a)"
        );
    }

    #[test]
    fn rewrite_table_name_quotes_when_needed() {
        assert_eq!(
            rewrite_create_table_name("CREATE TABLE t(a)", "new name").unwrap(),
            "CREATE TABLE \"new name\"(a)"
        );
        // A keyword new name is quoted so the rewritten sql re-parses.
        assert_eq!(
            rewrite_create_table_name("CREATE TABLE t(a)", "select").unwrap(),
            "CREATE TABLE \"select\"(a)"
        );
    }

    #[test]
    fn rewrite_table_name_handles_schema_qualifier() {
        // The name after the `.` is rewritten; the schema qualifier is left alone.
        assert_eq!(
            rewrite_create_table_name("CREATE TABLE main.t(a)", "t2").unwrap(),
            "CREATE TABLE main.t2(a)"
        );
    }

    // --- rewrite_index_table_name ---------------------------------------------

    #[test]
    fn rewrite_index_target_basic() {
        assert_eq!(
            rewrite_index_table_name("CREATE INDEX i ON t(a)", "t2").unwrap(),
            "CREATE INDEX i ON t2(a)"
        );
        assert_eq!(
            rewrite_index_table_name("CREATE UNIQUE INDEX i ON t (a, b) WHERE a > 0", "t2").unwrap(),
            "CREATE UNIQUE INDEX i ON t2 (a, b) WHERE a > 0"
        );
    }

    #[test]
    fn rewrite_index_target_handles_schema_qualifier_and_quoting() {
        assert_eq!(
            rewrite_index_table_name("CREATE INDEX i ON main.t(a)", "t2").unwrap(),
            "CREATE INDEX i ON main.t2(a)"
        );
        assert_eq!(
            rewrite_index_table_name("CREATE INDEX i ON t(a)", "t 2").unwrap(),
            "CREATE INDEX i ON \"t 2\"(a)"
        );
    }

    // --- rewrite_trigger_table_name -------------------------------------------

    #[test]
    fn rewrite_trigger_target_basic() {
        assert_eq!(
            rewrite_trigger_table_name(
                "CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END",
                "t2"
            )
            .unwrap(),
            "CREATE TRIGGER trg AFTER INSERT ON t2 BEGIN SELECT 1; END"
        );
    }

    #[test]
    fn rewrite_trigger_target_instead_of_uses_of_not_on() {
        // `INSTEAD OF` uses the keyword `OF`, not `ON`, so the first `ON` is still the
        // target table.
        assert_eq!(
            rewrite_trigger_table_name(
                "CREATE TRIGGER trg INSTEAD OF INSERT ON t BEGIN SELECT 1; END",
                "t2"
            )
            .unwrap(),
            "CREATE TRIGGER trg INSTEAD OF INSERT ON t2 BEGIN SELECT 1; END"
        );
    }

    #[test]
    fn rewrite_trigger_target_update_of_column_not_mistaken_for_target() {
        // `UPDATE OF a` uses `OF`; the `a` after it must NOT be taken as the target,
        // and the real target `t` (after the first `ON`) is the one rewritten.
        assert_eq!(
            rewrite_trigger_table_name(
                "CREATE TRIGGER trg AFTER UPDATE OF a ON t BEGIN SELECT 1; END",
                "t2"
            )
            .unwrap(),
            "CREATE TRIGGER trg AFTER UPDATE OF a ON t2 BEGIN SELECT 1; END"
        );
    }

    #[test]
    fn rewrite_trigger_target_handles_schema_qualifier() {
        // Only the name after the `.` changes; the schema qualifier is left alone.
        assert_eq!(
            rewrite_trigger_table_name(
                "CREATE TRIGGER trg AFTER INSERT ON main.t BEGIN SELECT 1; END",
                "t2"
            )
            .unwrap(),
            "CREATE TRIGGER trg AFTER INSERT ON main.t2 BEGIN SELECT 1; END"
        );
    }

    #[test]
    fn rewrite_trigger_target_quotes_when_needed() {
        assert_eq!(
            rewrite_trigger_table_name(
                "CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END",
                "t 2"
            )
            .unwrap(),
            "CREATE TRIGGER trg AFTER INSERT ON \"t 2\" BEGIN SELECT 1; END"
        );
    }

    #[test]
    fn rewrite_trigger_target_leaves_a_body_join_on_untouched() {
        // A join `ON` inside the `BEGIN...END` body comes AFTER `ON <table>`, so only
        // the target table is rewritten — the body's `ON x.id = y.id` is preserved.
        let sql =
            "CREATE TRIGGER trg AFTER INSERT ON t BEGIN INSERT INTO log SELECT x.id FROM x JOIN y ON x.id = y.id; END";
        let want =
            "CREATE TRIGGER trg AFTER INSERT ON t2 BEGIN INSERT INTO log SELECT x.id FROM x JOIN y ON x.id = y.id; END";
        assert_eq!(rewrite_trigger_table_name(sql, "t2").unwrap(), want);
    }

    // --- rename_column_in_create ----------------------------------------------

    #[test]
    fn rename_column_renames_the_definition_name() {
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b)", "b", "c").unwrap(),
            "CREATE TABLE t(a, c)"
        );
        // Case-insensitive match on `from`; the def keeps the `to` spelling verbatim.
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, B TEXT)", "b", "c").unwrap(),
            "CREATE TABLE t(a, c TEXT)"
        );
    }

    #[test]
    fn rename_column_updates_table_constraints_and_generated_exprs() {
        // Every reference within the column-list/constraint region is renamed: the
        // column def, a table-level PRIMARY KEY / UNIQUE / CHECK / FOREIGN KEY naming
        // it, and a generated-column expression referencing it.
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b, PRIMARY KEY(b))", "b", "c").unwrap(),
            "CREATE TABLE t(a, c, PRIMARY KEY(c))"
        );
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b, UNIQUE(b))", "b", "c").unwrap(),
            "CREATE TABLE t(a, c, UNIQUE(c))"
        );
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b, CHECK(b > 0))", "b", "c").unwrap(),
            "CREATE TABLE t(a, c, CHECK(c > 0))"
        );
        assert_eq!(
            rename_column_in_create(
                "CREATE TABLE t(a, b, FOREIGN KEY(b) REFERENCES u(x))",
                "b",
                "c"
            )
            .unwrap(),
            "CREATE TABLE t(a, c, FOREIGN KEY(c) REFERENCES u(x))"
        );
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b, g AS (b + a))", "b", "c").unwrap(),
            "CREATE TABLE t(a, c, g AS (c + a))"
        );
    }

    #[test]
    fn rename_column_leaves_the_table_name_and_string_literals_alone() {
        // A table named the same as the column is NOT renamed (it is before the `(`),
        // and a string literal that spells the column name is not an identifier.
        assert_eq!(
            rename_column_in_create("CREATE TABLE b(a, b)", "b", "c").unwrap(),
            "CREATE TABLE b(a, c)"
        );
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b, CHECK(a <> 'b'))", "b", "c").unwrap(),
            "CREATE TABLE t(a, c, CHECK(a <> 'b'))"
        );
    }

    #[test]
    fn rename_column_quotes_when_needed_and_matches_quoted_source() {
        // A keyword / spaced target is quoted so the result re-parses.
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, b)", "b", "select").unwrap(),
            "CREATE TABLE t(a, \"select\")"
        );
        // A quoted source column name is matched by its decoded value.
        assert_eq!(
            rename_column_in_create("CREATE TABLE t(a, \"my col\" INT)", "my col", "c").unwrap(),
            "CREATE TABLE t(a, c INT)"
        );
    }

    #[test]
    fn rename_column_result_reparses_with_the_new_name() {
        // Property: the rewrite re-parses and the renamed column is present under `to`.
        let out = rename_column_in_create("CREATE TABLE t(a INT, b TEXT, UNIQUE(b))", "b", "c")
            .unwrap();
        let ast = minisqlite_sql::parse(&out)
            .unwrap_or_else(|e| panic!("rewritten sql {out:?} must re-parse: {e}"));
        let ct = match ast.statements.as_slice() {
            [minisqlite_sql::Statement::CreateTable(ct)] => ct,
            other => panic!("expected one CREATE TABLE, got {other:?}"),
        };
        let cols = match &ct.body {
            minisqlite_sql::CreateTableBody::Columns { columns, .. } => columns,
            other => panic!("expected a column-list body, got {other:?}"),
        };
        assert_eq!(cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["a", "c"]);
    }

    // --- rename_column_in_index -----------------------------------------------

    #[test]
    fn rename_column_in_index_rewrites_columns_and_partial_where() {
        assert_eq!(
            rename_column_in_index("CREATE INDEX i ON t(b)", "b", "c").unwrap(),
            "CREATE INDEX i ON t(c)"
        );
        assert_eq!(
            rename_column_in_index("CREATE UNIQUE INDEX i ON t(a, b) WHERE b > 0", "b", "c")
                .unwrap(),
            "CREATE UNIQUE INDEX i ON t(a, c) WHERE c > 0"
        );
    }

    #[test]
    fn rename_column_in_index_leaves_index_and_table_names_alone() {
        // The index's own name and its target table are before the column region, so a
        // column that shares either name is not confused with them.
        assert_eq!(
            rename_column_in_index("CREATE INDEX b ON b(b)", "b", "c").unwrap(),
            "CREATE INDEX b ON b(c)"
        );
        // A schema-qualified target table is preserved; only the column list changes.
        assert_eq!(
            rename_column_in_index("CREATE INDEX i ON main.t(b)", "b", "c").unwrap(),
            "CREATE INDEX i ON main.t(c)"
        );
    }

    // --- remove_column_from_create --------------------------------------------

    #[test]
    fn remove_column_middle_first_and_last() {
        // Middle column: the item and its TRAILING comma go, one ", " stays.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b, c)", "b").unwrap(),
            "CREATE TABLE t(a, c)"
        );
        // First column: its trailing comma goes.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b, c)", "a").unwrap(),
            "CREATE TABLE t(b, c)"
        );
        // Last column: its LEADING comma goes.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b, c)", "c").unwrap(),
            "CREATE TABLE t(a, b)"
        );
    }

    #[test]
    fn remove_column_is_case_insensitive_and_keeps_types_and_defaults() {
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a INT, B TEXT DEFAULT 'x, y', c)", "b")
                .unwrap(),
            "CREATE TABLE t(a INT, c)"
        );
        // The comma inside the string literal must not be mistaken for a separator.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a INT, b TEXT DEFAULT 'x, y')", "a").unwrap(),
            "CREATE TABLE t(b TEXT DEFAULT 'x, y')"
        );
    }

    #[test]
    fn remove_column_with_nested_parens_in_other_columns() {
        // A CHECK/type with its own parens on a NEIGHBOUR column must not confuse the
        // depth-1 item scan.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a DECIMAL(10,2), b, c INT CHECK(c > 0))", "b")
                .unwrap(),
            "CREATE TABLE t(a DECIMAL(10,2), c INT CHECK(c > 0))"
        );
    }

    #[test]
    fn remove_column_preserves_trailing_table_constraints_and_options() {
        // Dropping the last COLUMN when a table constraint follows: the constraint is
        // the next depth-1 item, so the column takes its trailing comma and the
        // constraint stays put.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b, CHECK(a > 0))", "b").unwrap(),
            "CREATE TABLE t(a, CHECK(a > 0))"
        );
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b, PRIMARY KEY(a))", "b").unwrap(),
            "CREATE TABLE t(a, PRIMARY KEY(a))"
        );
        // Trailing options after the `)` are untouched.
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b) WITHOUT ROWID", "b").unwrap(),
            "CREATE TABLE t(a) WITHOUT ROWID"
        );
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, b) STRICT", "b").unwrap(),
            "CREATE TABLE t(a) STRICT"
        );
    }

    #[test]
    fn remove_column_handles_quoted_name() {
        assert_eq!(
            remove_column_from_create("CREATE TABLE t(a, \"my col\" INT, c)", "my col").unwrap(),
            "CREATE TABLE t(a, c)"
        );
    }

    #[test]
    fn remove_column_errors_on_missing_or_only_column() {
        assert!(remove_column_from_create("CREATE TABLE t(a, b)", "z").is_err());
        // The only column: refuse rather than produce `CREATE TABLE t()`.
        assert!(remove_column_from_create("CREATE TABLE t(a)", "a").is_err());
        // A column named like a table constraint's lead keyword is not confused: only
        // the true column def leads with an identifier.
        assert!(remove_column_from_create("CREATE TABLE t(a, CHECK(a > 0))", "check").is_err());
    }

    #[test]
    fn removed_column_result_reparses_with_one_fewer_column() {
        // Property: after removing a column the result re-parses and the column set is
        // exactly the original minus the dropped name, order preserved.
        for (create, drop, want) in [
            ("CREATE TABLE t(a INT, b TEXT, c REAL)", "b", vec!["a", "c"]),
            ("CREATE TABLE t(a, b, c, UNIQUE(a))", "c", vec!["a", "b"]),
            ("CREATE TABLE t(a, b CHECK(b > 0), c)", "b", vec!["a", "c"]),
        ] {
            let out = remove_column_from_create(create, drop).unwrap();
            let ast = minisqlite_sql::parse(&out)
                .unwrap_or_else(|e| panic!("rewritten sql {out:?} must re-parse: {e}"));
            let ct = match ast.statements.as_slice() {
                [minisqlite_sql::Statement::CreateTable(ct)] => ct,
                other => panic!("expected one CREATE TABLE, got {other:?}"),
            };
            let cols = match &ct.body {
                minisqlite_sql::CreateTableBody::Columns { columns, .. } => columns,
                other => panic!("expected a column-list body, got {other:?}"),
            };
            assert_eq!(cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), want);
        }
    }

    // --- quote_ident_if_needed ------------------------------------------------

    #[test]
    fn quoting_rules() {
        assert_eq!(quote_ident_if_needed("plain"), "plain");
        assert_eq!(quote_ident_if_needed("_underscore1"), "_underscore1");
        assert_eq!(quote_ident_if_needed("has space"), "\"has space\"");
        assert_eq!(quote_ident_if_needed("1leadingdigit"), "\"1leadingdigit\"");
        assert_eq!(quote_ident_if_needed(""), "\"\"");
        // Embedded double quote is doubled.
        assert_eq!(quote_ident_if_needed("a\"b"), "\"a\"\"b\"");
        // Keywords are always quoted (reserved or not) so re-parsing is unambiguous.
        assert_eq!(quote_ident_if_needed("select"), "\"select\"");
        assert_eq!(quote_ident_if_needed("TABLE"), "\"TABLE\"");
    }

    #[test]
    fn rewritten_name_is_reparseable_and_round_trips_the_value() {
        // Property-ish check across a set of adversarial names: after rewriting a
        // CREATE TABLE name to X, re-tokenizing must yield a single name token that
        // decodes back to exactly X (so the def's name will equal X on reload).
        for name in ["t2", "new name", "select", "a\"b", "_x9", "Δ", "with)paren"] {
            let sql = rewrite_create_table_name("CREATE TABLE t(a)", name).unwrap();
            let ast = minisqlite_sql::parse(&sql)
                .unwrap_or_else(|e| panic!("rewritten sql {sql:?} must re-parse: {e}"));
            match ast.statements.as_slice() {
                [minisqlite_sql::Statement::CreateTable(ct)] => {
                    assert_eq!(ct.name.name, name, "name must round-trip for {name:?} via {sql:?}");
                }
                other => panic!("expected one CREATE TABLE, got {other:?}"),
            }
        }
    }
}
