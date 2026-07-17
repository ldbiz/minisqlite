//! Conformance battery: SQLite **identifier and quoting syntax**.
//!
//! Every assertion here is transcribed from the SQLite documentation in
//! `spec/sqlite-doc/`, never from whatever the engine currently returns — a
//! failing case is the intended signal that the engine diverges from the spec.
//! Assertions are NEVER weakened to make the suite pass; a real divergence stays
//! spec-correct. Only a case that HANGS/aborts the process may be `#[ignore]`-d.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_keywords.html` — the four ways to quote an identifier and the
//!     historical quoting-rule exceptions:
//!       - `"x"` double quotes: an identifier (SQL standard).
//!       - `` `x` `` grave accents / backticks (ASCII 96): an identifier
//!         (MySQL compatibility).
//!       - `[x]` square brackets: an identifier (MS Access / SQL Server
//!         compatibility; not standard SQL).
//!       - `'x'` single quotes: a *string literal*, never an identifier.
//!       - A keyword enclosed in any of the three identifier-quote forms is a
//!         legal identifier.
//!   * `quirks.html` §8 "Double-quoted String Literals Are Accepted" — the DQS
//!     misfeature: SQLite "will also interpret a double-quotes string as string
//!     literal if it does not match any valid identifier." This is the *library*
//!     default (it can be disabled at compile time with `-DSQLITE_DQS=0`, and the
//!     `sqlite3` CLI disables it by default as of 3.41.0, but the library
//!     accepts it by default).
//!   * `quirks.html` §9 "Keywords Can Often Be Used As Identifiers" — many
//!     keywords may be used *unquoted* as identifiers in a context where it is
//!     clear an identifier is intended; e.g. `CREATE TABLE tableZ(INTEGER PRIMARY
//!     KEY)` makes a column named "INTEGER" (with no datatype).
//!   * `lang_naming.html` — "Like other SQL identifiers, schema names are
//!     case-insensitive": SQL identifiers are compared case-insensitively (ASCII
//!     folding), independent of the quoting form.
//!   * `c3ref/column_name.html` — a result column's name is the value of its "AS"
//!     clause; *without* an AS clause the name is UNSPECIFIED. Hence the
//!     column-name assertions below only pin AS-aliased columns; the
//!     case-insensitive *lookup* is what the unquoted-name cases pin.
//!
//! Each case is its own small `#[test]` (usually one behavioral assertion) so an
//! unsupported quoting form fails exactly that case rather than masking the rest.

mod conformance;
use conformance::*;

// ===========================================================================
// 1. Unquoted identifiers are CASE-INSENSITIVE.
//    (lang_naming.html: "identifiers are case-insensitive"; quirks.html §9.)
//    The invariant pinned here is the LOOKUP: an object created with one casing
//    resolves when referenced with any other casing. The reported *bare* column
//    name is unspecified (c3ref/column_name.html) and is NOT asserted here.
// ===========================================================================

#[test]
fn unquoted_table_name_resolves_case_insensitively() {
    // Only the table-name casing varies (Foo vs FOO); column stays `bar`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE Foo(bar INT)");
    exec(&mut db, "INSERT INTO Foo(bar) VALUES(1)");
    assert_scalar(&mut db, "SELECT bar FROM FOO", int(1));
}

#[test]
fn unquoted_column_name_resolves_case_insensitively() {
    // Only the column-name casing varies (Bar vs BAR); table stays `t`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(Bar INT)");
    exec(&mut db, "INSERT INTO t(Bar) VALUES(1)");
    assert_scalar(&mut db, "SELECT BAR FROM t", int(1));
}

#[test]
fn unquoted_all_positions_case_insensitive() {
    // The canonical example: declare Foo(Bar), write via foo(bar), read via FOO/BAR.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE Foo(Bar INT)");
    exec(&mut db, "INSERT INTO foo(bar) VALUES(1)");
    assert_scalar(&mut db, "SELECT BAR FROM FOO", int(1));
}

#[test]
fn unquoted_column_case_insensitive_in_where() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE person(Name TEXT, Age INT)");
    exec(&mut db, "INSERT INTO person(Name, Age) VALUES('a', 30)");
    assert_scalar(&mut db, "SELECT name FROM PERSON WHERE AGE = 30", text("a"));
}

#[test]
fn unquoted_insert_upper_select_lower_roundtrips() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE data(Val TEXT)");
    exec(&mut db, "INSERT INTO DATA(VAL) VALUES('hi')");
    assert_scalar(&mut db, "SELECT val FROM data", text("hi"));
}

// ===========================================================================
// 2. Result column names — c3ref/column_name.html.
//    "The name of a result column is the value of the 'AS' clause for that
//    column, if there is an AS clause. If there is no AS clause then the name of
//    the column is unspecified..." So we pin ONLY AS-aliased names (which are
//    specified and case-preserving). Bare-reference names are deliberately not
//    pinned, because the spec calls them unspecified.
// ===========================================================================

#[test]
fn column_name_from_as_alias_preserves_case() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE foo(bar INT)");
    // The AS value "Renamed" is the column name verbatim, including its casing.
    assert_columns(&mut db, "SELECT bar AS Renamed FROM foo", &["Renamed"]);
}

#[test]
fn column_name_from_quoted_as_alias_preserves_spaces_and_case() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE foo(bar INT)");
    // A double-quoted AS alias preserves spaces and exact spelling.
    assert_columns(&mut db, "SELECT bar AS \"My Col\" FROM foo", &["My Col"]);
}

#[test]
fn column_name_from_bracket_as_alias() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE foo(bar INT)");
    assert_columns(&mut db, "SELECT bar AS [Alias X] FROM foo", &["Alias X"]);
}

// ===========================================================================
// 3. Double-quoted identifiers — lang_keywords.html ("A keyword in double-quotes
//    is an identifier"). A quoted identifier preserves its exact spelling and may
//    contain spaces, keywords, or (via a doubled quote) a literal quote.
// ===========================================================================

#[test]
fn double_quoted_identifier_with_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(\"col a\" INT)");
    exec(&mut db, "INSERT INTO t(\"col a\") VALUES(5)");
    assert_scalar(&mut db, "SELECT \"col a\" FROM t", int(5));
}

#[test]
fn double_quoted_identifier_lookup_is_case_insensitive() {
    // Identifier comparison is ASCII-case-insensitive even for quoted names
    // (lang_naming.html: identifiers are case-insensitive) — "COL A" matches the
    // declared "col a".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(\"col a\" INT)");
    exec(&mut db, "INSERT INTO t(\"col a\") VALUES(5)");
    assert_scalar(&mut db, "SELECT \"COL A\" FROM t", int(5));
}

#[test]
fn double_quoted_keyword_columns() {
    // "select" and "from" are keywords; quoting makes them legal column names.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(\"select\" INT, \"from\" INT)");
    exec(&mut db, "INSERT INTO w(\"select\", \"from\") VALUES(1, 2)");
    assert_rows(&mut db, "SELECT \"select\", \"from\" FROM w", &[vec![int(1), int(2)]]);
}

#[test]
fn double_quoted_keyword_table_name() {
    // "order" is a keyword; quoting makes it a legal table name.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE \"order\"(x INT)");
    exec(&mut db, "INSERT INTO \"order\"(x) VALUES(3)");
    assert_scalar(&mut db, "SELECT x FROM \"order\"", int(3));
}

#[test]
fn double_quoted_table_name_with_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE \"my table\"(a INT)");
    exec(&mut db, "INSERT INTO \"my table\"(a) VALUES(8)");
    assert_scalar(&mut db, "SELECT a FROM \"my table\"", int(8));
}

#[test]
fn double_quoted_embedded_quote_is_doubled() {
    // Inside a double-quoted identifier, "" encodes one literal ", so this column
    // is named  a"b .
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(\"a\"\"b\" INT)");
    exec(&mut db, "INSERT INTO t(\"a\"\"b\") VALUES(4)");
    assert_scalar(&mut db, "SELECT \"a\"\"b\" FROM t", int(4));
}

// ===========================================================================
// 4. Bracket-quoted identifiers — lang_keywords.html ("A keyword enclosed in
//    square brackets is an identifier ... used by MS Access and SQL Server").
//    Brackets have no escape mechanism: content runs to the first ']'.
// ===========================================================================

#[test]
fn bracket_quoted_identifier_with_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u([weird name] INT)");
    exec(&mut db, "INSERT INTO u([weird name]) VALUES(6)");
    assert_scalar(&mut db, "SELECT [weird name] FROM u", int(6));
}

#[test]
fn bracket_quoted_keyword_column() {
    // "table" is a keyword; brackets make it a legal column name.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE bt([table] INT)");
    exec(&mut db, "INSERT INTO bt([table]) VALUES(11)");
    assert_scalar(&mut db, "SELECT [table] FROM bt", int(11));
}

#[test]
fn bracket_quoted_table_name() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE [my tab](a INT)");
    exec(&mut db, "INSERT INTO [my tab](a) VALUES(12)");
    assert_scalar(&mut db, "SELECT a FROM [my tab]", int(12));
}

#[test]
fn bracket_quoted_identifier_lookup_is_case_insensitive() {
    // Identifier folding is independent of the delimiter: [WEIRD NAME] resolves
    // the declared [weird name] (lang_naming.html — identifiers case-insensitive).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u([weird name] INT)");
    exec(&mut db, "INSERT INTO u([weird name]) VALUES(6)");
    assert_scalar(&mut db, "SELECT [WEIRD NAME] FROM u", int(6));
}

#[test]
fn bracket_identifier_has_no_escape_may_contain_open_bracket() {
    // Bracket quoting (MS Access / SQL Server) has no escape mechanism: content
    // runs to the FIRST ']', so [a[b] is one column named  a[b  — the inner '['
    // is ordinary text, not a nested delimiter.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t3([a[b] INT)");
    exec(&mut db, "INSERT INTO t3([a[b]) VALUES(13)");
    assert_scalar(&mut db, "SELECT [a[b] FROM t3", int(13));
}

// ===========================================================================
// 5. Backtick-quoted identifiers — lang_keywords.html ("A keyword enclosed in
//    grave accents (ASCII code 96) is an identifier ... used by MySQL"). A
//    doubled backtick escapes one, mirroring the double-quote rule. The literal
//    backticks below are valid, unescaped characters in a Rust string.
// ===========================================================================

#[test]
fn backtick_quoted_identifier_with_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE v(`b c` INT)");
    exec(&mut db, "INSERT INTO v(`b c`) VALUES(7)");
    assert_scalar(&mut db, "SELECT `b c` FROM v", int(7));
}

#[test]
fn backtick_quoted_keyword_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE bq(`select` INT)");
    exec(&mut db, "INSERT INTO bq(`select`) VALUES(9)");
    assert_scalar(&mut db, "SELECT `select` FROM bq", int(9));
}

#[test]
fn backtick_quoted_table_name() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE `my tbl`(a INT)");
    exec(&mut db, "INSERT INTO `my tbl`(a) VALUES(10)");
    assert_scalar(&mut db, "SELECT a FROM `my tbl`", int(10));
}

#[test]
fn backtick_embedded_backtick_is_doubled() {
    // A doubled backtick encodes one literal backtick, so this column is  a`b .
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(`a``b` INT)");
    exec(&mut db, "INSERT INTO t2(`a``b`) VALUES(4)");
    assert_scalar(&mut db, "SELECT `a``b` FROM t2", int(4));
}

#[test]
fn backtick_quoted_identifier_lookup_is_case_insensitive() {
    // Identifier folding is independent of the delimiter: `B C` resolves the
    // declared `b c` (lang_naming.html — identifiers case-insensitive).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE v(`b c` INT)");
    exec(&mut db, "INSERT INTO v(`b c`) VALUES(7)");
    assert_scalar(&mut db, "SELECT `B C` FROM v", int(7));
}

// ===========================================================================
// 6. Double-quoted string literal fallback (DQS) — quirks.html §8.
//    "SQLite will also interpret a double-quotes string as string literal if it
//    does not match any valid identifier." The library default accepts this, so
//    a double-quoted token with no matching identifier is TEXT. But when the
//    token DOES match an identifier, the identifier wins (it is only a fallback).
//
//    The misfeature applies in EVERY expression position (SELECT list, WHERE,
//    VALUES, DEFAULT — the CLI even splits it into DQS_DML vs DQS_DDL), so the
//    "unresolved -> string" cases below spread across positions. The
//    "identifier wins" cases pin that the fallback only fires on a resolution
//    failure: when a matching column exists, the identifier is used, not the text.
// ===========================================================================

#[test]
fn dqs_unresolved_double_quote_is_string_literal() {
    // No column/table named `hello` exists, so "hello" falls back to TEXT 'hello'.
    let mut db = mem();
    assert_scalar(&mut db, "SELECT \"hello\"", text("hello"));
}

#[test]
fn dqs_unresolved_double_quote_typeof_is_text() {
    let mut db = mem();
    assert_scalar(&mut db, "SELECT typeof(\"hello\")", text("text"));
}

#[test]
fn dqs_matching_double_quote_reads_the_column() {
    // "c" matches column c, so it is read as the column (value 42), not as text.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(c INT)");
    exec(&mut db, "INSERT INTO t(c) VALUES(42)");
    assert_scalar(&mut db, "SELECT \"c\" FROM t", int(42));
}

#[test]
fn dqs_identifier_wins_over_string_fallback() {
    // Even a "word-like" name resolves as the column when one exists: "hello"
    // reads column hello (9), it is NOT the string 'hello'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(hello INT)");
    exec(&mut db, "INSERT INTO t(hello) VALUES(9)");
    assert_scalar(&mut db, "SELECT \"hello\" FROM t", int(9));
}

#[test]
fn dqs_unresolved_in_where_is_string_literal() {
    // DQS is not SELECT-list-only: with no column named `lit`, "lit" in WHERE is
    // the string 'lit', so the row (x = 'lit') matches.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x TEXT)");
    exec(&mut db, "INSERT INTO t(x) VALUES('lit')");
    assert_scalar(&mut db, "SELECT x FROM t WHERE x = \"lit\"", text("lit"));
}

#[test]
fn dqs_unresolved_in_insert_values_is_string_literal() {
    // DQS_DML: a double-quoted VALUES token with no matching identifier is TEXT.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x TEXT)");
    exec(&mut db, "INSERT INTO t(x) VALUES(\"lit\")");
    assert_scalar(&mut db, "SELECT x FROM t", text("lit"));
}

#[test]
fn dqs_unresolved_in_column_default_is_string_literal() {
    // DQS_DDL: a double-quoted DEFAULT token with no matching identifier is TEXT,
    // so the defaulted column holds 'lit'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INT, x TEXT DEFAULT \"lit\")");
    exec(&mut db, "INSERT INTO t(id) VALUES(1)");
    assert_scalar(&mut db, "SELECT x FROM t", text("lit"));
}

#[test]
fn dqs_ambiguous_double_quote_errors_and_does_not_fall_back() {
    // quirks.html §8: the fallback fires ONLY when the double-quoted token "does not
    // match any valid identifier". Column `x` exists in BOTH `a` and `b`, so `"x"` DOES
    // match a valid identifier (an ambiguous one) — real sqlite raises "ambiguous column
    // name: x"; it must NOT silently become the string literal 'x'. This is the not-found
    // vs. ambiguous distinction: an ambiguity is still a match, so the identifier path wins
    // and errors.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x INTEGER)");
    exec(&mut db, "CREATE TABLE b(x INTEGER)");
    exec(&mut db, "INSERT INTO a(x) VALUES(1)");
    exec(&mut db, "INSERT INTO b(x) VALUES(2)");
    let e = assert_query_error(&mut db, "SELECT \"x\" FROM a, b");
    assert!(
        format!("{e:?}").contains("ambiguous"),
        "expected an ambiguous-column error (not a silent string-literal fallback), got {e:?}"
    );
}

#[test]
fn dqs_qualified_double_quote_never_falls_back() {
    // Only a BARE `"name"` participates in the DQS fallback. A table-qualified
    // `t."name"` is ALWAYS an identifier, so an unresolved one is a hard error — it never
    // becomes the string literal 'nope'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "INSERT INTO t(x) VALUES(1)");
    assert_query_error(&mut db, "SELECT t.\"nope\" FROM t");
}

#[test]
fn bracket_and_backtick_unresolved_names_do_not_fall_back() {
    // The DQS misfeature is DOUBLE-QUOTE ONLY (quirks.html §8). A bracket `[nope]` or
    // backtick `` `nope` `` identifier that resolves to no column is a hard error, never a
    // string literal — only `"nope"` would fall back.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT [nope]");
    assert_query_error(&mut db, "SELECT `nope`");
}

#[test]
fn dqs_double_quoted_boolean_keyword_is_text_not_bool() {
    // A double-quoted `"true"`/`"false"` is NOT the boolean literal (value 1/0). The
    // parser keeps the TRUE/FALSE keyword shortcut to *bare* (unquoted) tokens, so a
    // quoted one stays a bare name that — resolving to no column — falls back to the
    // DQS text literal. Its typeof is 'text' and its value is the spelling, not 1/0.
    // (Dropping that unquoted-only gate would wrongly turn `"true"` into integer 1.)
    eval_eq("\"true\"", text("true"));
    eval_eq("typeof(\"true\")", text("text"));
    eval_eq("\"false\"", text("false"));
}

#[test]
fn dqs_unresolved_in_aggregate_query_is_string_literal() {
    // Grouping-context path: the aggregate `max(x)` turns on grouping, so the bare
    // `"nope"` in the same SELECT is bound through the grouping guard. It names no
    // column, so it must NOT raise "must appear in GROUP BY" / "no such column" — it
    // falls back to the DQS text literal 'nope' (a constant is legal beside an
    // aggregate). Removing the `from_dqs` bail in the grouping guard would make this
    // query error instead of returning the row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "INSERT INTO t(x) VALUES(3)");
    exec(&mut db, "INSERT INTO t(x) VALUES(5)");
    assert_rows(&mut db, "SELECT max(x), \"nope\" FROM t", &[vec![int(5), text("nope")]]);
}

#[test]
fn dqs_in_order_by_simple_sorts_by_constant_but_compound_errors() {
    // lang_select.html §4 (ORDER BY, rules 1-3 and the compound-handling paragraph):
    // a SIMPLE SELECT's ORDER BY may be "any arbitrary expression", but a COMPOUND
    // SELECT's ORDER BY term that is not an integer ordinal or an output-column alias
    // "must be exactly the same as an expression used as an output column" — otherwise
    // "it is an error".
    //
    // An unresolved bare `"nope"` becomes the DQS text constant 'nope'. In the simple
    // SELECT that is a legal (constant) sort key, so the row is returned; in the
    // compound SELECT the same constant is neither an ordinal, an alias, nor an output
    // expression, so real sqlite raises an error. The two ORDER BY paths therefore
    // treat the identical DQS token differently ON PURPOSE — a future change must NOT
    // make the compound matcher fall back and silently accept the constant.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "INSERT INTO t(x) VALUES(1)");
    assert_scalar(&mut db, "SELECT x FROM t ORDER BY \"nope\"", int(1));
    assert_query_error(&mut db, "SELECT x FROM t UNION SELECT x FROM t ORDER BY \"nope\"");
}

// ===========================================================================
// 7. Single quotes denote a string literal, not an identifier —
//    lang_keywords.html ("A keyword in single quotes is a string literal").
//    (lang_keywords.html also documents a historical exception: a single-quoted
//    token where an identifier is required but a string literal is NOT allowed is
//    read as an identifier. Every case below is an expression context where a
//    string literal IS allowed, so the string-literal reading holds.)
// ===========================================================================

#[test]
fn single_quote_is_string_literal() {
    eval_eq("'x'", text("x"));
}

#[test]
fn single_quote_typeof_is_text() {
    eval_eq("typeof('x')", text("text"));
}

#[test]
fn single_quoted_token_is_string_not_column() {
    // With a column named `bar`, 'bar' is STILL the literal string 'bar' for every
    // row — single quotes never denote an identifier.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE foo(bar INT)");
    exec(&mut db, "INSERT INTO foo(bar) VALUES(1)");
    assert_scalar(&mut db, "SELECT 'bar' FROM foo", text("bar"));
}

// ===========================================================================
// 8. Keywords used as UNQUOTED identifiers — quirks.html §9. Many keywords may be
//    used unquoted where the parse is unambiguous. The documented example makes a
//    column literally named "INTEGER".
//    (Probe/record: not every keyword is usable unquoted in every position.)
// ===========================================================================

#[test]
fn keyword_integer_as_unquoted_column_name() {
    // quirks.html §9: `tableZ(INTEGER PRIMARY KEY)` makes a column named INTEGER.
    // Here INTEGER is exercised as an identifier in three positions.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tableZ(INTEGER PRIMARY KEY)");
    exec(&mut db, "INSERT INTO tableZ(INTEGER) VALUES(7)");
    assert_scalar(&mut db, "SELECT INTEGER FROM tableZ", int(7));
}

// ---- join & window keywords as UNQUOTED identifiers -------------------------
//
// quirks.html §9 in its sharpest form. SQLite's tokenizer reclassifies the join
// keywords (NATURAL/LEFT/RIGHT/FULL/INNER/OUTER/CROSS — its `TK_JOIN_KW`) and the
// window keywords OVER/WINDOW to `TK_ID` in an identifier position, and resolves
// FILTER the same way by look-ahead (`tokenize.c`). So each is a legal UNQUOTED
// column name, table name, and `AS` alias — only its grammatical position (an
// operator slot) makes it a join/window keyword. These pin that the engine treats
// them as identifiers where an identifier is intended, while joins/windows below
// still parse. (Was a hard "expected an identifier"/"expected an expression"
// parse error before the reserved set was narrowed to match SQLite.)

#[test]
fn join_keywords_are_unquoted_column_names() {
    // A tree-node table naming its children `left`/`right` — the canonical case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE node(id INTEGER, left INTEGER, right INTEGER, full TEXT)");
    exec(&mut db, "INSERT INTO node VALUES (1, 2, 3, 'root')");
    assert_rows(
        &mut db,
        "SELECT id, left, right, full FROM node",
        &[vec![int(1), int(2), int(3), text("root")]],
    );
    // Qualified and in an expression / WHERE / UPDATE too.
    assert_scalar(&mut db, "SELECT node.left + node.right FROM node", int(5));
    assert_scalar(&mut db, "SELECT full FROM node WHERE left = 2", text("root"));
    exec(&mut db, "UPDATE node SET left = 20 WHERE right = 3");
    assert_scalar(&mut db, "SELECT left FROM node", int(20));
}

#[test]
fn window_keywords_are_unquoted_column_names() {
    // OVER, FILTER, WINDOW as ordinary column names.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE cfg(over INTEGER, filter TEXT, window INTEGER)");
    exec(&mut db, "INSERT INTO cfg VALUES (1, 'f', 9)");
    assert_rows(&mut db, "SELECT over, filter, window FROM cfg", &[vec![int(1), text("f"), int(9)]]);
}

#[test]
fn join_keyword_as_unquoted_table_name() {
    // A table literally named `cross` (and one named `outer`).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE cross(x INTEGER)");
    exec(&mut db, "INSERT INTO cross(x) VALUES (7)");
    assert_scalar(&mut db, "SELECT x FROM cross", int(7));
    // A `table.*` on the keyword-named table resolves too.
    assert_rows(&mut db, "SELECT cross.* FROM cross", &[vec![int(7)]]);
}

#[test]
fn keyword_as_alias_forms() {
    // `AS <keyword>` names a result column with a join/window keyword.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (2, 3)");
    assert_columns(&mut db, "SELECT a AS left, b AS over FROM t", &["left", "over"]);
}

#[test]
fn keyword_column_names_do_not_break_joins() {
    // The reserved-set change must not break join parsing: a table with keyword
    // column names still joins normally, and the join operators still parse.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(id INTEGER, v TEXT)");
    exec(&mut db, "CREATE TABLE b(id INTEGER, w TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (1,'a'),(2,'b')");
    exec(&mut db, "INSERT INTO b VALUES (2,'B'),(3,'C')");
    assert_rows(
        &mut db,
        "SELECT a.id, b.id FROM a LEFT JOIN b ON a.id = b.id ORDER BY a.id",
        &[vec![int(1), null()], vec![int(2), int(2)]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM a CROSS JOIN b", int(4));
    // Aliased tables around a join with a keyword-named join operator.
    assert_rows(
        &mut db,
        "SELECT l.id FROM a l INNER JOIN b r ON l.id = r.id",
        &[vec![int(2)]],
    );
}

#[test]
fn window_clause_still_parses_with_window_as_a_name() {
    // A trailing WINDOW definition clause must not be swallowed as the FROM table's
    // alias now that WINDOW is a legal identifier.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(x INTEGER)");
    exec(&mut db, "INSERT INTO w VALUES (1),(2),(3)");
    assert_rows(
        &mut db,
        "SELECT x, row_number() OVER win FROM w WINDOW win AS (ORDER BY x) ORDER BY x",
        &[vec![int(1), int(1)], vec![int(2), int(2)], vec![int(3), int(3)]],
    );
}

#[test]
fn fallback_keywords_each_generated_always_are_column_names() {
    // EACH / GENERATED / ALWAYS are `%fallback ID` keywords in SQLite (parse.y), so
    // they are legal unquoted column names. (Was a hard parse error before the reserved
    // set was narrowed.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER, each INTEGER, generated INTEGER, always INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 3, 4)");
    assert_rows(&mut db, "SELECT each, generated, always FROM t", &[vec![int(2), int(3), int(4)]]);
}

#[test]
fn each_as_name_does_not_break_for_each_row_trigger() {
    // `FOR EACH ROW` still parses (EACH read positionally as the keyword), even though
    // EACH is otherwise a usable identifier.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(m TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t FOR EACH ROW BEGIN INSERT INTO log VALUES ('hit'); END",
    );
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_scalar(&mut db, "SELECT m FROM log", text("hit"));
}

#[test]
fn generated_as_name_does_not_break_generated_column() {
    // `GENERATED ALWAYS AS (…)` still parses and computes, even though GENERATED and
    // ALWAYS are usable identifiers elsewhere.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g(a INTEGER, b GENERATED ALWAYS AS (a * 2), c AS (a + 1))");
    exec(&mut db, "INSERT INTO g(a) VALUES (10)");
    assert_rows(&mut db, "SELECT a, b, c FROM g", &[vec![int(10), int(20), int(11)]]);
}
