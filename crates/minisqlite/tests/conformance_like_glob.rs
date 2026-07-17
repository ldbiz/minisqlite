//! Conformance: the LIKE, GLOB, and ESCAPE pattern-matching operators, plus their
//! `like()` / `glob()` function forms.
//!
//! Every expectation here is transcribed from the SQLite documentation, NOT from
//! what this engine returns:
//!   * `spec/sqlite-doc/lang_expr.html` §5 ("The LIKE, GLOB, REGEXP, MATCH, and
//!     extract operators") — the operator semantics.
//!   * `spec/sqlite-doc/lang_corefunc.html` — the `glob(X,Y)` / `like(X,Y[,Z])`
//!     function forms (argument order is REVERSED vs the infix operator: X is the
//!     pattern, Y is the string).
//!
//! Documented contract exercised below:
//!   * LIKE: `%` matches any run of zero-or-more characters, `_` matches exactly
//!     one character, all other characters match case-INSENSITIVELY but only over
//!     ASCII (`'a' LIKE 'A'` is true; `'æ' LIKE 'Æ'` is false).
//!   * ESCAPE: the escape character makes the following `%`, `_`, or escape
//!     character a literal.
//!   * GLOB: Unix globbing — `*`, `?`, `[...]` — and is case-SENSITIVE. §5 defers
//!     the wildcard grammar to "Unix file globbing syntax" rather than spelling it
//!     out, so the `[...]` character-class cases below (a set, an inclusive `a-c`
//!     range, a leading `^` negation, and a leading `]` as a literal member) are
//!     transcribed from that standard Unix-glob behavior, which real SQLite follows.
//!   * NOT LIKE / NOT GLOB invert the sense; a NULL operand yields NULL (not 0/1).
//!
//! Each case asserts the spec-correct value. If the engine disagrees, the assertion
//! is LEFT failing rather than weakened to pass. Only a case that hangs or
//! panics the engine may be `#[ignore]`d, in its own test, with a reason.

mod conformance;

use conformance::*;

// ---- LIKE: the `%` wildcard (any run, including empty) -----------------------
// lang_expr.html §5: 'A percent symbol ("%") in the LIKE pattern matches any
// sequence of zero or more characters in the string.'

#[test]
fn like_percent_prefix_suffix_and_middle() {
    eval_eq("'abc' LIKE 'a%'", int(1));
    eval_eq("'abc' LIKE '%c'", int(1));
    eval_eq("'abc' LIKE '%b%'", int(1));
    eval_eq("'abc' LIKE 'a%c'", int(1));
}

#[test]
fn like_percent_matches_empty_run() {
    // `%` matches zero characters, so the interior `%` spans nothing here.
    eval_eq("'ac' LIKE 'a%c'", int(1));
    // A lone `%` matches any whole string, including the empty string.
    eval_eq("'abc' LIKE '%'", int(1));
    eval_eq("'' LIKE '%'", int(1));
    // Adjacent `%` collapse to the same "any run" meaning.
    eval_eq("'abc' LIKE '%%'", int(1));
}

#[test]
fn like_percent_interior_does_not_overmatch() {
    // An interior `%` matches a run, but the literals around it must still line up:
    // the trailing `d` has nothing to match in "abc", so the whole pattern fails.
    eval_eq("'abc' LIKE 'a%d'", int(0));
    // ...and it succeeds when a real `d` follows the spanned run.
    eval_eq("'abcd' LIKE 'a%d'", int(1));
}

// ---- LIKE: the `_` wildcard (exactly one character) --------------------------
// lang_expr.html §5: 'An underscore ("_") ... matches any single character.'

#[test]
fn like_underscore_matches_exactly_one() {
    eval_eq("'abc' LIKE 'a_c'", int(1));
    eval_eq("'abc' LIKE 'a__'", int(1));
    eval_eq("'abc' LIKE '___'", int(1));
    eval_eq("'a' LIKE '_'", int(1));
}

#[test]
fn like_underscore_wrong_length_fails() {
    // `_` consumes one character and no more/less than one.
    eval_eq("'abc' LIKE 'a_'", int(0));
    eval_eq("'abc' LIKE '____'", int(0));
    eval_eq("'' LIKE '_'", int(0));
}

// ---- LIKE: literal / exact and empty-pattern matching ------------------------

#[test]
fn like_exact_literal_match() {
    eval_eq("'abc' LIKE 'abc'", int(1));
    eval_eq("'abc' LIKE 'abd'", int(0));
    eval_eq("'abc' LIKE 'abcd'", int(0));
}

#[test]
fn like_empty_pattern_matches_only_empty_string() {
    eval_eq("'' LIKE ''", int(1));
    eval_eq("'a' LIKE ''", int(0));
}

// ---- LIKE: ASCII case-insensitivity ------------------------------------------
// lang_expr.html §5: 'Any other character matches itself or its lower/upper case
// equivalent (i.e. case-insensitive matching).' and the example '"a" LIKE "A" is
// TRUE'.

#[test]
fn like_is_ascii_case_insensitive_both_directions() {
    eval_eq("'a' LIKE 'A'", int(1));
    eval_eq("'A' LIKE 'a'", int(1));
    eval_eq("'ABC' LIKE 'abc'", int(1));
    eval_eq("'abc' LIKE 'ABC'", int(1));
}

#[test]
fn like_case_insensitivity_combines_with_wildcards() {
    eval_eq("'abc' LIKE 'AB%'", int(1));
    eval_eq("'abc' LIKE 'A_C'", int(1));
    eval_eq("'Hello' LIKE 'h_llo'", int(1));
}

#[test]
fn like_non_ascii_is_case_sensitive() {
    // lang_expr.html §5 example: SQLite folds case for ASCII only, so
    // 'æ' LIKE 'Æ' is FALSE even though 'a' LIKE 'A' is TRUE.
    eval_eq("'æ' LIKE 'Æ'", int(0));
    // The identical non-ASCII character still matches itself.
    eval_eq("'æ' LIKE 'æ'", int(1));
}

// ---- NOT LIKE ----------------------------------------------------------------
// lang_expr.html §5: 'Both GLOB and LIKE may be preceded by the NOT keyword to
// invert the sense of the test.'

#[test]
fn not_like_inverts_the_match() {
    eval_eq("'abc' NOT LIKE 'x%'", int(1));
    eval_eq("'abc' NOT LIKE 'a%'", int(0));
    eval_eq("'abc' NOT LIKE 'abc'", int(0));
    eval_eq("'abc' NOT LIKE 'abd'", int(1));
}

// ---- LIKE with NULL operands -------------------------------------------------
// A NULL operand makes the result NULL (three-valued logic), not 0 or 1.

#[test]
fn like_with_null_operand_is_null() {
    eval_eq("NULL LIKE 'a'", null());
    eval_eq("'a' LIKE NULL", null());
    eval_eq("NULL LIKE NULL", null());
}

#[test]
fn not_like_with_null_operand_is_null() {
    // NULL propagates through the NOT as well.
    eval_eq("NULL NOT LIKE 'a'", null());
    eval_eq("'a' NOT LIKE NULL", null());
}

// ---- LIKE ... ESCAPE ---------------------------------------------------------
// lang_expr.html §5: 'The escape character followed by a percent symbol (%),
// underscore (_), or a second instance of the escape character itself matches a
// literal percent symbol, underscore, or a single escape character, respectively.'
//
// SQL string literals do not treat backslash specially, so `'\'` is a one-byte
// string. In these Rust source literals a single SQL backslash is written `\\`.

#[test]
fn like_escape_makes_percent_literal() {
    // Pattern `a\%c` with ESCAPE `\` means the literal string "a%c".
    eval_eq("'a%c' LIKE 'a\\%c' ESCAPE '\\'", int(1));
    // ...so a real character where the literal `%` is expected does NOT match.
    eval_eq("'axc' LIKE 'a\\%c' ESCAPE '\\'", int(0));
}

#[test]
fn like_escape_makes_underscore_literal() {
    // Pattern `a\_c` with ESCAPE `\` means the literal string "a_c".
    eval_eq("'a_c' LIKE 'a\\_c' ESCAPE '\\'", int(1));
    eval_eq("'axc' LIKE 'a\\_c' ESCAPE '\\'", int(0));
}

#[test]
fn like_escape_makes_escape_char_literal() {
    // `\\` (escape + escape) matches a single literal backslash.
    // SQL: 'a\c' LIKE 'a\\c' ESCAPE '\'
    eval_eq("'a\\c' LIKE 'a\\\\c' ESCAPE '\\'", int(1));
}

#[test]
fn like_escape_realistic_percent_suffix() {
    // A common real-world use: match a value that literally ends in '%'.
    // SQL: '100%' LIKE '100\%' ESCAPE '\'
    eval_eq("'100%' LIKE '100\\%' ESCAPE '\\'", int(1));
    eval_eq("'100x' LIKE '100\\%' ESCAPE '\\'", int(0));
}

#[test]
fn like_escape_character_is_not_hardcoded_to_backslash() {
    // ESCAPE takes any single character; here '#' escapes the '%'.
    // SQL: 'a%c' LIKE 'a#%c' ESCAPE '#'
    eval_eq("'a%c' LIKE 'a#%c' ESCAPE '#'", int(1));
    eval_eq("'axc' LIKE 'a#%c' ESCAPE '#'", int(0));
}

#[test]
fn like_without_escape_treats_percent_as_wildcard() {
    // Contrast with the ESCAPE cases above: unescaped `%` is still a wildcard.
    eval_eq("'axc' LIKE 'a%c'", int(1));
}

// ---- LIKE vs GLOB have distinct wildcard vocabularies ------------------------
// lang_expr.html §5 gives LIKE exactly two wildcards (`%` and `_`); every "other
// character matches itself". So GLOB's `*` and `?` are ORDINARY characters to LIKE
// (and, symmetrically, LIKE's `%`/`_` are ordinary to GLOB — see the GLOB section
// below). A matcher that shared a single wildcard vocabulary across both operators
// would pass every per-operator case yet fail these.

#[test]
fn like_treats_glob_wildcards_as_literals() {
    eval_eq("'a*c' LIKE 'a*c'", int(1));
    eval_eq("'axc' LIKE 'a*c'", int(0));
    eval_eq("'a?c' LIKE 'a?c'", int(1));
    eval_eq("'axc' LIKE 'a?c'", int(0));
}

// ---- GLOB: `*` and `?` wildcards, case sensitivity ---------------------------
// lang_expr.html §5: 'The GLOB operator ... uses the Unix file globbing syntax
// for its wildcards. Also, GLOB is case sensitive, unlike LIKE.'

#[test]
fn glob_star_wildcard() {
    eval_eq("'abc' GLOB 'a*'", int(1));
    eval_eq("'abc' GLOB '*c'", int(1));
    eval_eq("'abc' GLOB '*b*'", int(1));
    eval_eq("'abc' GLOB '*'", int(1));
    eval_eq("'' GLOB '*'", int(1));
}

#[test]
fn glob_star_matches_empty_run_at_any_position() {
    // `*` matches zero characters, so a leading/trailing/interior `*` can span
    // nothing while the literal parts still match exactly.
    eval_eq("'abc' GLOB 'abc*'", int(1));
    eval_eq("'abc' GLOB '*abc'", int(1));
    eval_eq("'abc' GLOB 'a*bc'", int(1));
}

#[test]
fn glob_question_matches_exactly_one() {
    eval_eq("'abc' GLOB 'a?c'", int(1));
    eval_eq("'abc' GLOB '???'", int(1));
    eval_eq("'abc' GLOB '??'", int(0));
    eval_eq("'abc' GLOB '????'", int(0));
}

#[test]
fn glob_exact_literal_match() {
    eval_eq("'abc' GLOB 'abc'", int(1));
    eval_eq("'abc' GLOB 'abd'", int(0));
}

#[test]
fn glob_is_case_sensitive() {
    // Unlike LIKE, GLOB does not fold case.
    eval_eq("'abc' GLOB 'A*'", int(0));
    eval_eq("'ABC' GLOB 'abc'", int(0));
    eval_eq("'abc' GLOB 'ABC'", int(0));
    eval_eq("'ABC' GLOB 'ABC'", int(1));
}

#[test]
fn glob_treats_like_wildcards_as_literals() {
    // Symmetric to `like_treats_glob_wildcards_as_literals`: LIKE's `%` and `_`
    // have no special meaning to GLOB — they match themselves.
    eval_eq("'a%c' GLOB 'a%c'", int(1));
    eval_eq("'axc' GLOB 'a%c'", int(0));
    eval_eq("'a_c' GLOB 'a_c'", int(1));
    eval_eq("'axc' GLOB 'a_c'", int(0));
}

// ---- GLOB: `[...]` character classes -----------------------------------------
// Unix glob character classes: a set, an inclusive range `a-c`, or a leading `^`
// to negate the set.

#[test]
fn glob_character_class_set() {
    eval_eq("'b' GLOB '[abc]'", int(1));
    eval_eq("'d' GLOB '[abc]'", int(0));
}

#[test]
fn glob_character_class_range() {
    eval_eq("'abc' GLOB '[a-c]bc'", int(1));
    eval_eq("'b' GLOB '[a-c]'", int(1));
    eval_eq("'d' GLOB '[a-c]'", int(0));
    eval_eq("'5' GLOB '[0-9]'", int(1));
    // Ranges are case-sensitive too.
    eval_eq("'M' GLOB '[a-z]'", int(0));
}

#[test]
fn glob_character_class_negation() {
    eval_eq("'abc' GLOB '[^x]bc'", int(1));
    eval_eq("'x' GLOB '[^a-c]'", int(1));
    eval_eq("'a' GLOB '[^a-c]'", int(0));
    eval_eq("'a' GLOB '[^0-9]'", int(1));
}

#[test]
fn glob_character_class_leading_bracket_is_literal() {
    // Standard Unix-glob bracket rule: a `]` immediately after `[` is a literal
    // member, not the terminator — so `[]]` is the one-character class "]".
    eval_eq("']' GLOB '[]]'", int(1));
    eval_eq("'x' GLOB '[]]'", int(0));
}

// ---- NOT GLOB ----------------------------------------------------------------

#[test]
fn not_glob_inverts_the_match() {
    eval_eq("'abc' NOT GLOB 'x*'", int(1));
    eval_eq("'abc' NOT GLOB 'a*'", int(0));
}

// ---- GLOB with NULL operands -------------------------------------------------

#[test]
fn glob_with_null_operand_is_null() {
    eval_eq("NULL GLOB 'a*'", null());
    eval_eq("'a' GLOB NULL", null());
    eval_eq("NULL NOT GLOB 'a*'", null());
}

// ---- Function forms: like(X, Y) / glob(X, Y) ---------------------------------
// lang_corefunc.html: `like(X,Y)` == `Y LIKE X`, `glob(X,Y)` == `Y GLOB X`; the
// FIRST argument is the pattern and the SECOND is the string (reversed vs infix).
// PROBE: if the engine has not wired these forms, the call errors and the case
// fails (rather than being weakened to pass), surfacing the discrepancy.

#[test]
fn like_function_form_two_args() {
    // like('a%','abc') == 'abc' LIKE 'a%'
    eval_eq("like('a%','abc')", int(1));
    eval_eq("like('%c','abc')", int(1));
    eval_eq("like('x%','abc')", int(0));
    // Case-insensitivity carries through the function form.
    eval_eq("like('A%','abc')", int(1));
}

#[test]
fn glob_function_form_two_args() {
    // glob('a*','abc') == 'abc' GLOB 'a*'
    eval_eq("glob('a*','abc')", int(1));
    eval_eq("glob('*c','abc')", int(1));
    eval_eq("glob('x*','abc')", int(0));
    // Case-sensitivity carries through the function form.
    eval_eq("glob('A*','abc')", int(0));
}

#[test]
fn function_form_argument_order_is_pattern_then_string() {
    // Pins the reversed order: with the args swapped the pattern becomes the
    // literal string 'abc' and the "string" becomes 'a%' / 'a*', which do not
    // match, so a naive same-order implementation would return 1 here.
    eval_eq("like('abc','a%')", int(0));
    eval_eq("glob('abc','a*')", int(0));
}

#[test]
fn like_function_form_three_args_escape() {
    // like(X, Y, Z) == 'Y' LIKE 'X' ESCAPE 'Z'; here X='a\%c', Y='a%c', Z='\'.
    // SQL: like('a\%c','a%c','\')
    eval_eq("like('a\\%c','a%c','\\')", int(1));
    // The escaped '%' is literal, so a wildcard-style match must NOT succeed.
    eval_eq("like('a\\%c','axc','\\')", int(0));
}

#[test]
fn function_form_with_null_operand_is_null() {
    eval_eq("like('a%',NULL)", null());
    eval_eq("like(NULL,'abc')", null());
    eval_eq("glob('a*',NULL)", null());
    eval_eq("glob(NULL,'abc')", null());
}

// ---- Single-character wildcards match one Unicode character (not one byte) ----
// lang_expr.html §5: `_` "matches any single character"; GLOB's `?` (Unix glob)
// likewise. A "character" is a Unicode scalar, so one multi-byte character is a
// single match — a byte-wise matcher would treat `é`/`æ` as several positions and
// fail these even though every ASCII `_`/`?` case still passed.

#[test]
fn single_char_wildcards_match_one_unicode_character() {
    eval_eq("'æ' LIKE '_'", int(1));
    eval_eq("'é' GLOB '?'", int(1));
    // A run of single-character wildcards counts characters, not bytes: "café" is
    // four characters, so `ca__` / `ca??` match it exactly.
    eval_eq("'café' LIKE 'ca__'", int(1));
    eval_eq("'café' GLOB 'ca??'", int(1));
}
