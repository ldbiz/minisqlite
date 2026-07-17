//! Conformance battery for the built-in scalar STRING functions.
//!
//! Every expectation is transcribed from the SQLite documentation —
//! `spec/sqlite-doc/lang_corefunc.html` §3 for the function contracts, and
//! `spec/sqlite-doc/printf.html` for the exact `format()`/`printf()`
//! substitutions — NOT from what this engine happens to return. When the engine
//! disagrees with the spec, the spec-correct assertion stays (and fails) rather
//! than being weakened to match the engine.
//!
//! `format()`/`printf()` is documented to "work like ... the printf() function
//! from the standard C library" (`lang_corefunc.html` §3), so its width /
//! precision / flag details follow the C-printf conventions that `printf.html`
//! spells out (e.g. default float precision 6, a signed >=2-digit `%e` exponent).
//! The two spec pages conflict on ONE point — `%p` is "an alias for %X"
//! (upper-case) in `lang_corefunc.html` §3 but "works like %x" (lower-case) in
//! `printf.html` — and the more specific `printf.html` substitution table is
//! followed; see the `%p` case below.
//!
//! Assertions go through the pinned facade `minisqlite::Connection` via the shared
//! harness helper `eval_eq(expr, Value)` (`SELECT <expr>` in a throwaway in-memory
//! database, compared with the harness's class-sensitive `value_eq`). `Value` has
//! no `PartialEq`, so all comparisons are the harness constructors + `eval_eq`.

mod conformance;

use conformance::*;

// ===========================================================================
// length(X)   —   lang_corefunc.html#length
// "For a string value X ... the number of Unicode code points (not bytes) ...
//  prior to the first U+0000. For a blob value X ... the number of bytes ...
//  If X is NULL then length(X) is NULL. If X is numeric then length(X) returns
//  the length of a string representation of X."
// ===========================================================================

#[test]
fn length_of_text() {
    eval_eq("length('abc')", int(3));
    eval_eq("length('')", int(0));
    eval_eq("length('hello world')", int(11));
}

#[test]
fn length_of_null_is_null() {
    eval_eq("length(NULL)", null());
}

#[test]
fn length_of_number_uses_its_text_form() {
    // A numeric X is measured by the length of its string representation.
    eval_eq("length(123)", int(3));
    eval_eq("length(-123)", int(4)); // the '-' counts
    eval_eq("length(12.5)", int(4)); // "12.5"
}

#[test]
fn length_of_blob_counts_bytes() {
    eval_eq("length(x'0102')", int(2));
    eval_eq("length(x'00ff10')", int(3));
}

#[test]
fn length_of_text_counts_code_points_not_bytes() {
    // "café" is 4 code points but 5 UTF-8 bytes ('é' is 2 bytes): length counts
    // characters, so it is 4 (the length/octet_length distinction).
    eval_eq("length('café')", int(4));
    eval_eq("length('áéíóú')", int(5));
}

// ===========================================================================
// octet_length(X)   —   lang_corefunc.html#octet_length
// "the number of bytes in the encoding of text string X. If X is NULL ... NULL.
//  If X is a BLOB ... same as length(X). If X is a numeric value ... the number
//  of bytes in a text rendering of that number."
// ===========================================================================

#[test]
fn octet_length_basic() {
    eval_eq("octet_length('abc')", int(3));
    eval_eq("octet_length('')", int(0));
}

#[test]
fn octet_length_of_null_is_null() {
    eval_eq("octet_length(NULL)", null());
}

#[test]
fn octet_length_counts_bytes_of_the_utf8_encoding() {
    // Byte count, not character count. In the default (UTF-8) encoding, 'é' is 2
    // bytes, so "café" is 5 bytes even though length() is 4. The spec notes the
    // answer is encoding-dependent; this pins the default UTF-8 database.
    eval_eq("octet_length('café')", int(5));
}

#[test]
fn octet_length_of_blob_and_number() {
    eval_eq("octet_length(x'0102')", int(2)); // same as length for a BLOB
    eval_eq("octet_length(123)", int(3)); // bytes of the text rendering "123"
}

// ===========================================================================
// substr(X,Y,Z) / substr(X,Y) / substring(...)   —   lang_corefunc.html#substr
// "begins with the Y-th character and which is Z characters long. If Z is
//  omitted ... all characters through the end ... beginning with the Y-th. The
//  left-most character of X is number 1. If Y is negative ... counting from the
//  right. If Z is negative then the abs(Z) characters preceding the Y-th
//  character are returned. If X is a string then indices count UTF code points.
//  If X is a BLOB then the indices count bytes." substring is an alias.
// ===========================================================================

#[test]
fn substr_three_arg_positive() {
    eval_eq("substr('abcdef',2,3)", text("bcd"));
    eval_eq("substr('abcdef',1,3)", text("abc")); // left-most char is 1
    eval_eq("substr('abcdef',4,2)", text("de"));
}

#[test]
fn substr_two_arg_runs_to_end() {
    eval_eq("substr('abcdef',2)", text("bcdef"));
    eval_eq("substr('abcdef',3)", text("cdef"));
    eval_eq("substr('abcdef',1)", text("abcdef"));
}

#[test]
fn substring_is_an_alias_for_substr() {
    eval_eq("substring('abcdef',2,3)", text("bcd"));
    eval_eq("substring('abcdef',2)", text("bcdef"));
}

#[test]
fn substr_negative_y_counts_from_the_right() {
    eval_eq("substr('abcdef',-2,2)", text("ef"));
    eval_eq("substr('abcdef',-2)", text("ef")); // to the end, from 2nd-from-right
    eval_eq("substr('abcdef',-3,2)", text("de"));
}

#[test]
fn substr_negative_z_takes_characters_before_y() {
    // abs(Z) characters PRECEDING the Y-th character (not including it).
    eval_eq("substr('abcdef',3,-2)", text("ab")); // 2 chars before 'c'
    eval_eq("substr('abcdef',2,-1)", text("a")); // 1 char before 'b'
}

#[test]
fn substr_boundary_positions() {
    eval_eq("substr('abc',0)", text("abc")); // Y=0: from the very start
    eval_eq("substr('abcdef',2,100)", text("bcdef")); // Z past the end clamps
    eval_eq("substr('abcdef',10)", text("")); // Y past the end is empty
}

#[test]
fn substr_null_arguments_yield_null() {
    eval_eq("substr(NULL,1,1)", null());
    eval_eq("substr('abc',NULL)", null());
    eval_eq("substr('abc',1,NULL)", null());
}

#[test]
fn substr_of_text_indexes_code_points() {
    // Multi-byte characters are indexed by code point, not byte.
    eval_eq("substr('áéíóú',2,2)", text("éí"));
    eval_eq("substr('áéíóú',-1)", text("ú"));
}

#[test]
fn substr_of_blob_indexes_bytes_and_returns_blob() {
    eval_eq("substr(x'00112233',2,2)", blob(&[0x11, 0x22]));
    eval_eq("substr(x'00112233',2)", blob(&[0x11, 0x22, 0x33]));
    eval_eq("substr(x'0102030405',-2,2)", blob(&[0x04, 0x05]));
}

// ===========================================================================
// upper(X) / lower(X)   —   lang_corefunc.html#upper / #lower
// "all ASCII characters converted to lower case ... works for ASCII characters
//  only." (upper is the mirror.)
// ===========================================================================

#[test]
fn upper_and_lower_basic() {
    eval_eq("upper('abcXYZ')", text("ABCXYZ"));
    eval_eq("lower('abcXYZ')", text("abcxyz"));
    eval_eq("upper('abc123')", text("ABC123")); // digits unchanged
}

#[test]
fn upper_and_lower_are_ascii_only() {
    // Non-ASCII letters are left as-is by the default (non-ICU) build.
    eval_eq("upper('café')", text("CAFé"));
    eval_eq("lower('CAFÉ')", text("cafÉ"));
}

#[test]
fn upper_and_lower_of_null_is_null() {
    eval_eq("upper(NULL)", null());
    eval_eq("lower(NULL)", null());
}

// ===========================================================================
// trim(X[,Y]) / ltrim / rtrim   —   lang_corefunc.html#trim / #ltrim / #rtrim
// "removing any and all characters that appear in Y from both ends of X. If the
//  Y argument is omitted, trim(X) removes spaces ..." (l/r trim one side).
// ===========================================================================

#[test]
fn trim_family_default_removes_spaces() {
    eval_eq("trim('  ab  ')", text("ab"));
    eval_eq("ltrim('  ab')", text("ab"));
    eval_eq("rtrim('ab  ')", text("ab"));
    // Each one-sided form leaves the other side alone.
    eval_eq("ltrim('  ab  ')", text("ab  "));
    eval_eq("rtrim('  ab  ')", text("  ab"));
}

#[test]
fn trim_family_two_arg_uses_char_set() {
    // Y is a SET of characters; any of them is stripped, greedily.
    eval_eq("trim('xxabxx','x')", text("ab"));
    eval_eq("ltrim('xxab','x')", text("ab"));
    eval_eq("rtrim('abxx','x')", text("ab"));
    eval_eq("trim('xyzabczyx','xyz')", text("abc"));
    eval_eq("trim('abacaba','ab')", text("c"));
}

#[test]
fn trim_only_touches_the_ends() {
    // Interior members of the set are preserved.
    eval_eq("trim('xaxbx','x')", text("axb"));
    eval_eq("trim('   ')", text("")); // all spaces removed
}

#[test]
fn trim_family_null_yields_null() {
    eval_eq("trim(NULL)", null());
    eval_eq("trim(NULL,'x')", null());
    eval_eq("ltrim(NULL)", null());
    eval_eq("rtrim(NULL)", null());
}

// ===========================================================================
// replace(X,Y,Z)   —   lang_corefunc.html#replace
// "substituting string Z for every occurrence of string Y in string X. The
//  BINARY collating sequence is used ... If Y is an empty string then return X
//  unchanged."
// ===========================================================================

#[test]
fn replace_basic() {
    eval_eq("replace('abcabc','b','X')", text("aXcaXc"));
    eval_eq("replace('abc','x','y')", text("abc")); // no occurrence -> unchanged
    eval_eq("replace('aaa','a','bb')", text("bbbbbb"));
    eval_eq("replace('abcabc','abc','X')", text("XX"));
}

#[test]
fn replace_empty_pattern_returns_x_unchanged() {
    // The empty-Y rule takes precedence over everything, including a NULL Z: the
    // documented "return X unchanged" is evaluated before Z is examined.
    eval_eq("replace('abcabc','','X')", text("abcabc"));
    eval_eq("replace('abc','',NULL)", text("abc"));
}

#[test]
fn replace_uses_binary_case_sensitive_matching() {
    // BINARY collation => case-sensitive: only lowercase 'a' matches.
    eval_eq("replace('aAaA','a','X')", text("XAXA"));
}

#[test]
fn replace_null_arguments_yield_null() {
    eval_eq("replace(NULL,'a','b')", null());
    eval_eq("replace('abc',NULL,'b')", null());
    eval_eq("replace('abc','a',NULL)", null());
}

// ===========================================================================
// instr(X,Y)   —   lang_corefunc.html#instr
// "finds the first occurrence of string Y within string X and returns the
//  number of prior characters plus 1, or 0 if Y is nowhere found within X ...
//  If either X or Y are NULL ... the result is NULL."  (both BLOB -> bytes.)
// ===========================================================================

#[test]
fn instr_finds_first_occurrence() {
    eval_eq("instr('abcabc','c')", int(3));
    eval_eq("instr('abcabc','bc')", int(2));
    eval_eq("instr('abcabc','abc')", int(1));
    eval_eq("instr('hello world','world')", int(7));
}

#[test]
fn instr_not_found_is_zero() {
    eval_eq("instr('abc','x')", int(0));
    eval_eq("instr('abc','abcd')", int(0)); // needle longer than haystack
}

#[test]
fn instr_empty_needle_is_one() {
    eval_eq("instr('abc','')", int(1));
}

#[test]
fn instr_null_argument_yields_null() {
    eval_eq("instr(NULL,'a')", null());
    eval_eq("instr('abc',NULL)", null());
}

#[test]
fn instr_counts_characters_for_text() {
    // "number of prior characters plus 1": 'b' is the 3rd code point of "aébc".
    eval_eq("instr('aébc','b')", int(3));
}

#[test]
fn instr_of_two_blobs_counts_bytes() {
    // Both arguments BLOB: "one more than the number of bytes prior".
    eval_eq("instr(x'0102030405',x'0304')", int(3));
    eval_eq("instr(x'0102030405',x'0909')", int(0));
}

// ===========================================================================
// concat(...) / concat_ws(SEP,...)   —   lang_corefunc.html#concat / #concat_ws
// concat: "concatenation of the string representation of all of its non-NULL
//  arguments. If all arguments are NULL, then concat() returns an empty string."
// concat_ws: joins non-null args after the first with SEP; "If the first
//  argument is NULL ... returns NULL. If all arguments other than the first are
//  NULL ... returns an empty string."
// ===========================================================================

#[test]
fn concat_joins_non_null_arguments() {
    eval_eq("concat('a','b','c')", text("abc"));
    eval_eq("concat('a',NULL,'c')", text("ac")); // NULLs contribute nothing
    eval_eq("concat(1,2,3)", text("123")); // string representation of numbers
}

#[test]
fn concat_of_all_null_is_empty_string() {
    eval_eq("concat(NULL,NULL)", text(""));
}

#[test]
fn concat_ws_joins_with_separator() {
    eval_eq("concat_ws('-','a','b')", text("a-b"));
    eval_eq("concat_ws(',','a',NULL,'c')", text("a,c")); // no separator around a skipped NULL
    eval_eq("concat_ws('-','a')", text("a"));
    eval_eq("concat_ws(',',1,2,3)", text("1,2,3"));
}

#[test]
fn concat_ws_null_separator_is_null() {
    eval_eq("concat_ws(NULL,'a','b')", null());
}

#[test]
fn concat_ws_all_values_null_is_empty_string() {
    eval_eq("concat_ws('-',NULL,NULL)", text(""));
}

// ===========================================================================
// hex(X)   —   lang_corefunc.html#hex
// "interprets its argument as a BLOB and returns ... the upper-case hexadecimal
//  rendering ... If X ... is an integer or floating point number ... the binary
//  number is first converted into a UTF8 text representation, then that text is
//  interpreted as a BLOB. Hence, 'hex(12345678)' renders as '3132333435363738'."
// hex(NULL) is intentionally NOT asserted: the spec does not document hex of a
// NULL argument, so there is no spec value to transcribe.
// ===========================================================================

#[test]
fn hex_of_text_and_blob() {
    eval_eq("hex('abc')", text("616263")); // 'a'=0x61 'b'=0x62 'c'=0x63
    eval_eq("hex(x'0f10')", text("0F10")); // upper-case output
    eval_eq("hex(x'00ff')", text("00FF"));
    eval_eq("hex('')", text(""));
}

#[test]
fn hex_of_number_uses_its_text_form() {
    // The spec's own worked example: the digits, not the binary integer.
    eval_eq("hex(12345678)", text("3132333435363738"));
}

// ===========================================================================
// quote(X)   —   lang_corefunc.html#quote
// "the text of an SQL literal which is the value of its argument ... Strings are
//  surrounded by single-quotes with escapes on interior quotes as needed. BLOBs
//  are encoded as hexadecimal literals."
// ===========================================================================

#[test]
fn quote_of_text_single_quotes_and_escapes() {
    eval_eq("quote('ab')", text("'ab'"));
    eval_eq("quote('a''b')", text("'a''b'")); // interior quote doubled
    eval_eq("quote('')", text("''"));
    eval_eq("quote('it''s')", text("'it''s'"));
}

#[test]
fn quote_of_null_and_integer() {
    eval_eq("quote(NULL)", text("NULL"));
    eval_eq("quote(123)", text("123"));
}

#[test]
fn quote_of_blob_is_a_hex_literal() {
    eval_eq("quote(x'0f10')", text("X'0F10'"));
}

// ===========================================================================
// char(X1,...,XN)   —   lang_corefunc.html#char
// "a string composed of characters having the unicode code point values of
//  integers X1 through XN."
// char(NULL) is intentionally NOT asserted: the spec does not document a NULL
// (non-integer) argument, so there is no spec value to transcribe.
// ===========================================================================

#[test]
fn char_builds_a_string_from_code_points() {
    eval_eq("char(72,73)", text("HI"));
    eval_eq("char(65)", text("A"));
    eval_eq("char(104,105)", text("hi"));
    eval_eq("char(65,66,67)", text("ABC"));
}

#[test]
fn char_supports_non_ascii_code_points() {
    eval_eq("char(233)", text("é")); // U+00E9
}

// ===========================================================================
// unicode(X)   —   lang_corefunc.html#unicode
// "the numeric unicode code point corresponding to the first character of the
//  string X."
// ===========================================================================

#[test]
fn unicode_returns_first_char_code_point() {
    eval_eq("unicode('A')", int(65));
    eval_eq("unicode('AB')", int(65)); // first character only
    eval_eq("unicode('a')", int(97));
    eval_eq("unicode('é')", int(233)); // U+00E9
}

#[test]
fn unicode_and_char_round_trip() {
    eval_eq("unicode(char(97))", int(97));
    eval_eq("char(unicode('Z'))", text("Z"));
}

// ===========================================================================
// format(FORMAT,...) / printf(FORMAT,...)   —   lang_corefunc.html#format,
// #printf and the substitution/flag/width/precision tables in printf.html.
// "If the FORMAT argument is missing or NULL then the result is NULL ... missing
//  arguments are assumed to have a NULL value, which is translated into 0 or 0.0
//  for numeric formats or an empty string for %s." printf() is an alias.
// ===========================================================================

#[test]
fn format_pinned_examples() {
    eval_eq("format('%d-%s',5,'x')", text("5-x"));
    eval_eq("printf('%d',42)", text("42"));
    eval_eq("format('%.2f',3.14159)", text("3.14"));
    eval_eq("format('%5d',3)", text("    3")); // width 5, right-justified
    eval_eq("printf('%x',255)", text("ff")); // lower-case hex
}

#[test]
fn format_integer_specifiers() {
    eval_eq("format('%i',42)", text("42")); // %i is an alias for %d
    eval_eq("format('%u',5)", text("5"));
    eval_eq("format('%X',255)", text("FF")); // upper-case hex
    eval_eq("format('%o',8)", text("10")); // octal
    // printf.html §2.2: the length modifier "is ignored for the format() SQL
    // function which always uses 64-bit values", so -1 as unsigned is 2^64 - 1.
    eval_eq("format('%u',-1)", text("18446744073709551615"));
}

#[test]
fn format_integer_sign_flags() {
    eval_eq("format('%+d',5)", text("+5"));
    eval_eq("format('%+d',-5)", text("-5"));
    eval_eq("format('% d',5)", text(" 5")); // space flag
}

#[test]
fn format_integer_width_and_precision() {
    eval_eq("format('%-5d',3)", text("3    ")); // left-justified
    eval_eq("format('%05d',42)", text("00042")); // zero-padded
    eval_eq("format('%05d',-42)", text("-0042"));
    eval_eq("format('%.4d',42)", text("0042")); // precision = min digits
}

#[test]
fn format_alternate_form_prefixes() {
    // "#" prepends 0/0x/0X for octal/hex.
    eval_eq("format('%#x',255)", text("0xff"));
    eval_eq("format('%#X',255)", text("0XFF"));
    eval_eq("format('%#o',8)", text("010"));
}

#[test]
fn format_string_specifier() {
    eval_eq("format('%s','hello')", text("hello"));
    eval_eq("format('%.2s','hello')", text("he")); // precision = leading bytes
    eval_eq("format('%5s','hi')", text("   hi"));
    eval_eq("format('%-5s','hi')", text("hi   "));
    eval_eq("format('[%s]',NULL)", text("[]")); // NULL arg -> empty string
    eval_eq("format('%z','zed')", text("zed")); // %z is interchangeable with %s
}

#[test]
fn format_percent_and_literals() {
    eval_eq("format('%%')", text("%"));
    eval_eq("format('100%%')", text("100%"));
    eval_eq("format('no substitutions')", text("no substitutions"));
}

#[test]
fn format_float_specifiers() {
    eval_eq("format('%f',3.5)", text("3.500000")); // default precision 6
    eval_eq("format('%.0f',3.7)", text("4")); // rounds
    eval_eq("format('%.3f',2.0)", text("2.000"));
    eval_eq("format('%e',1000.0)", text("1.000000e+03"));
    eval_eq("format('%E',1000.0)", text("1.000000E+03"));
}

#[test]
fn format_general_float() {
    eval_eq("format('%g',100000.0)", text("100000"));
    eval_eq("format('%g',1000000.0)", text("1e+06"));
}

#[test]
fn format_char_extracts_first_character_of_the_string() {
    // For the SQL format() function %c takes the FIRST CHARACTER of the (string)
    // argument (printf.html), NOT a C code point.
    eval_eq("format('%c','hello')", text("h"));
    eval_eq("format('%c','X')", text("X"));
    eval_eq("format('%.3c','x')", text("xxx")); // precision N>1 repeats
}

#[test]
fn format_sql_escape_specifiers() {
    // %q doubles single quotes; %Q also wraps in quotes and renders NULL as NULL.
    eval_eq("format('%q','a''b')", text("a''b"));
    eval_eq("format('%Q','a''b')", text("'a''b'"));
    eval_eq("format('%Q',NULL)", text("NULL"));
}

#[test]
fn format_comma_grouping() {
    eval_eq("format('%,d',1000)", text("1,000"));
    eval_eq("format('%,d',2147483647)", text("2,147,483,647"));
}

#[test]
fn format_percent_p_aliases_hex_and_n_is_ignored() {
    // printf.html substitution table: "%p ... works like %x" -> lower-case hex.
    // The two spec pages CONFLICT here: lang_corefunc.html §3 instead calls %p
    // "an alias for %X" (upper-case -> "FF"). The more specific printf.html table
    // is followed. If a differential run against real sqlite3 renders "FF", this
    // assertion (and the engine) would diverge on this one contradictory point.
    eval_eq("format('%p',255)", text("ff"));
    // lang_corefunc.html §3: "The %n format is silently ignored and does not
    // consume an argument" — so the following %d still reads 7, not a missing arg.
    eval_eq("format('a%nb%d',7)", text("ab7"));
}

#[test]
fn format_missing_arguments_default_to_zero_and_empty() {
    eval_eq("format('%d')", text("0")); // missing numeric arg -> 0
    eval_eq("format('%d %s %d',1)", text("1  0")); // missing %s -> "", missing %d -> 0
}

#[test]
fn format_null_format_string_is_null() {
    eval_eq("format(NULL)", null());
    eval_eq("printf(NULL)", null());
}
