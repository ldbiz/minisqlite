//! The `LIKE` and `GLOB` pattern matchers (lang_expr.html §5).
//!
//! Both are exposed as public functions so the eventual `like()`/`glob()` scalar
//! functions in `minisqlite-functions` reuse *this* matcher rather than growing a
//! second, drifting copy.
//!
//! Matching is bounded: the pattern is tokenized once, then matched against the
//! subject with the classic iterative wildcard algorithm (a single saved
//! backtrack point for the most recent `%`/`*`). That is O(n·m) worst case and
//! always terminates — there is no recursive descent that a pathological pattern
//! like `%%%%…a` could blow the stack or the clock on.
//!
//! Both operate on Unicode scalar values (`char`), because `_`/`?` match one
//! *character* and SQLite works in UTF-8; `LIKE` folds only ASCII `A-Z`/`a-z`
//! (SQLite is case-insensitive for ASCII only). Perf note: a pattern that is a
//! constant should be tokenized once at bind time rather than per row — a future
//! optimization the binder can layer on top of this matcher.

/// Does `text` match the `LIKE` `pattern`? `%` matches any run (including empty),
/// `_` matches exactly one character, matching is ASCII-case-insensitive, and an
/// `escape` character (if given) makes the following pattern character literal.
pub fn like_matches(text: &str, pattern: &str, escape: Option<char>) -> bool {
    let cfg = Cfg { all_wild: '%', one_wild: '_', classes: false, fold_case: true, escape };
    matches_with(text, pattern, &cfg)
}

/// Does `text` match the `GLOB` `pattern`? `*` matches any run (including empty),
/// `?` matches exactly one character, `[...]` is a character class (ranges `a-z`
/// and a leading `^` to negate), and matching is case-*sensitive*.
pub fn glob_matches(text: &str, pattern: &str) -> bool {
    let cfg = Cfg { all_wild: '*', one_wild: '?', classes: true, fold_case: false, escape: None };
    matches_with(text, pattern, &cfg)
}

/// The three axes on which `LIKE` and `GLOB` differ, so one matcher serves both.
struct Cfg {
    all_wild: char,
    one_wild: char,
    classes: bool,
    fold_case: bool,
    escape: Option<char>,
}

impl Cfg {
    /// Fold an ASCII upper-case letter to lower for case-insensitive `LIKE`;
    /// identity for `GLOB`. Only ASCII is folded (SQLite does not case-fold
    /// non-ASCII by default).
    fn fold(&self, c: char) -> char {
        if self.fold_case { c.to_ascii_lowercase() } else { c }
    }
}

/// A compiled pattern element. A `Class` is only produced for `GLOB`.
enum Tok {
    /// `%` / `*` — matches any run of characters.
    AllWild,
    /// `_` / `?` — matches exactly one character.
    OneWild,
    /// A literal character (already case-folded for `LIKE`).
    Lit(char),
    /// A `GLOB` `[...]` character class.
    Class { neg: bool, items: Vec<ClassItem> },
}

/// One member of a character class: a single char or an inclusive range.
enum ClassItem {
    Ch(char),
    Range(char, char),
}

fn matches_with(text: &str, pattern: &str, cfg: &Cfg) -> bool {
    let subject: Vec<char> = text.chars().collect();
    let toks = tokenize(pattern, cfg);
    match_tokens(&subject, &toks, cfg)
}

/// Compile a pattern string into tokens once. Escape handling and (for GLOB)
/// character-class parsing happen here, so the hot matching loop below is simple.
fn tokenize(pattern: &str, cfg: &Cfg) -> Vec<Tok> {
    let pat: Vec<char> = pattern.chars().collect();
    let mut toks = Vec::with_capacity(pat.len());
    let mut i = 0;
    while i < pat.len() {
        let c = pat[i];
        if cfg.escape == Some(c) {
            // The escape makes the *next* character literal. A trailing escape (no
            // following character) is treated as a literal escape character.
            if let Some(&next) = pat.get(i + 1) {
                toks.push(Tok::Lit(cfg.fold(next)));
                i += 2;
            } else {
                toks.push(Tok::Lit(cfg.fold(c)));
                i += 1;
            }
        } else if c == cfg.all_wild {
            toks.push(Tok::AllWild);
            i += 1;
        } else if c == cfg.one_wild {
            toks.push(Tok::OneWild);
            i += 1;
        } else if cfg.classes && c == '[' {
            match parse_class(&pat, i) {
                Some((tok, consumed)) => {
                    toks.push(tok);
                    i += consumed;
                }
                // An unterminated `[` is a literal `[` (SQLite treats a `[` with no
                // closing `]` as an ordinary character).
                None => {
                    toks.push(Tok::Lit('['));
                    i += 1;
                }
            }
        } else {
            toks.push(Tok::Lit(cfg.fold(c)));
            i += 1;
        }
    }
    toks
}

/// Parse a `GLOB` character class beginning at `pat[start] == '['`. Returns the
/// token and the number of pattern chars consumed (including both brackets), or
/// `None` if there is no closing `]` (so the caller can treat `[` as a literal).
///
/// Rules (lang_expr.html / Unix glob): a leading `^` negates; a `]` as the first
/// class member (right after `[` or `[^`) is a literal `]`; `a-z` is an inclusive
/// range, but a `-` that is first or last in the class is a literal `-`.
fn parse_class(pat: &[char], start: usize) -> Option<(Tok, usize)> {
    let mut i = start + 1;
    let mut neg = false;
    if pat.get(i) == Some(&'^') {
        neg = true;
        i += 1;
    }
    let mut items = Vec::new();
    // A `]` immediately here is a literal member, not the terminator.
    if pat.get(i) == Some(&']') {
        items.push(ClassItem::Ch(']'));
        i += 1;
    }
    while let Some(&c) = pat.get(i) {
        if c == ']' {
            return Some((Tok::Class { neg, items }, i + 1 - start));
        }
        // A range `c-d` only when the '-' is followed by a real member (not the
        // closing ']'); otherwise '-' is a literal member.
        if pat.get(i + 1) == Some(&'-') && pat.get(i + 2).is_some_and(|&d| d != ']') {
            let hi = pat[i + 2];
            items.push(ClassItem::Range(c, hi));
            i += 3;
        } else {
            items.push(ClassItem::Ch(c));
            i += 1;
        }
    }
    // Reached end of pattern with no closing ']'.
    None
}

/// Whether character `c` is a member of the class.
fn class_matches(neg: bool, items: &[ClassItem], c: char) -> bool {
    let mut found = false;
    for it in items {
        let hit = match it {
            ClassItem::Ch(x) => *x == c,
            // A reversed range (lo > hi) simply matches nothing.
            ClassItem::Range(lo, hi) => *lo <= c && c <= *hi,
        };
        if hit {
            found = true;
            break;
        }
    }
    found ^ neg
}

/// Does token `tok` match the single character `ch`? `AllWild` never matches here
/// (it is handled by the backtracking loop); every other token matches exactly one
/// character.
fn token_matches_single(tok: &Tok, ch: char, cfg: &Cfg) -> bool {
    match tok {
        Tok::AllWild => false,
        Tok::OneWild => true,
        Tok::Lit(lc) => *lc == cfg.fold(ch),
        Tok::Class { neg, items } => class_matches(*neg, items, ch),
    }
}

/// The bounded iterative wildcard match. `star`/`mark` remember the most recent
/// `AllWild` and how much of the subject it has so far absorbed; on a mismatch we
/// let that wildcard swallow one more character. Every backtrack advances `mark`,
/// so the loop runs at most `subject.len() * toks.len()` times and terminates.
fn match_tokens(subject: &[char], toks: &[Tok], cfg: &Cfg) -> bool {
    let mut si = 0;
    let mut ti = 0;
    let mut star: Option<usize> = None;
    let mut mark = 0;

    while si < subject.len() {
        if ti < toks.len() && token_matches_single(&toks[ti], subject[si], cfg) {
            si += 1;
            ti += 1;
        } else if ti < toks.len() && matches!(toks[ti], Tok::AllWild) {
            star = Some(ti);
            mark = si;
            ti += 1;
        } else if let Some(sp) = star {
            ti = sp + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    // Any tokens left must all be `%`/`*` (each matching the now-empty remainder).
    while ti < toks.len() && matches!(toks[ti], Tok::AllWild) {
        ti += 1;
    }
    ti == toks.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_basic_wildcards() {
        assert!(like_matches("abc", "abc", None));
        assert!(like_matches("abc", "a%", None));
        assert!(like_matches("abc", "%c", None));
        assert!(like_matches("abc", "a_c", None));
        assert!(like_matches("abc", "%", None));
        assert!(like_matches("", "%", None)); // % matches the empty run
        assert!(!like_matches("abc", "a_", None)); // _ is exactly one
        assert!(!like_matches("abc", "abcd", None));
        assert!(!like_matches("", "_", None)); // _ needs a character
    }

    #[test]
    fn like_is_ascii_case_insensitive() {
        assert!(like_matches("ABC", "abc", None));
        assert!(like_matches("abc", "ABC", None));
        assert!(like_matches("Hello", "h%O", None));
        // 'a' LIKE 'A' is TRUE but non-ASCII is case-sensitive (spec §5 example).
        assert!(like_matches("a", "A", None));
        assert!(!like_matches("æ", "Æ", None));
    }

    #[test]
    fn like_percent_backtracking() {
        // Multiple % and interior anchors — the classic backtracking cases.
        assert!(like_matches("xaxbxc", "%a%b%c", None));
        assert!(like_matches("aaa", "%a", None));
        assert!(like_matches("mississippi", "m%i%i%i%", None));
        assert!(!like_matches("mississippi", "m%x%", None));
    }

    #[test]
    fn like_escape() {
        // \% matches a literal %, \_ a literal _, \\ a literal \.
        assert!(like_matches("a%b", "a\\%b", Some('\\')));
        assert!(!like_matches("axb", "a\\%b", Some('\\'))); // \% is literal, not wildcard
        assert!(like_matches("a_b", "a\\_b", Some('\\')));
        assert!(like_matches("50%", "50\\%", Some('\\')));
        assert!(like_matches("a\\b", "a\\\\b", Some('\\'))); // escaped escape
        // An escaped ordinary character is just that character (still case-folded).
        assert!(like_matches("A", "\\a", Some('\\')));
    }

    #[test]
    fn glob_basic_wildcards() {
        assert!(glob_matches("abc", "abc"));
        assert!(glob_matches("abc", "a*"));
        assert!(glob_matches("abc", "*c"));
        assert!(glob_matches("abc", "a?c"));
        assert!(glob_matches("abc", "*"));
        assert!(!glob_matches("abc", "a?")); // ? is exactly one
    }

    #[test]
    fn glob_is_case_sensitive() {
        assert!(!glob_matches("ABC", "abc"));
        assert!(glob_matches("ABC", "ABC"));
        assert!(!glob_matches("abc", "A*"));
    }

    #[test]
    fn glob_character_classes() {
        assert!(glob_matches("b", "[abc]"));
        assert!(!glob_matches("d", "[abc]"));
        assert!(glob_matches("m", "[a-z]"));
        assert!(!glob_matches("M", "[a-z]")); // case-sensitive range
        assert!(glob_matches("5", "[0-9]"));
        assert!(glob_matches("foo9", "foo[0-9]"));
        // Negation with a leading ^ (SQLite's negation character).
        assert!(glob_matches("d", "[^abc]"));
        assert!(!glob_matches("a", "[^abc]"));
        assert!(glob_matches("x", "[^0-9]"));
        assert!(!glob_matches("7", "[^0-9]"));
    }

    #[test]
    fn glob_class_edge_cases() {
        // A leading `]` is a literal member; a first/last `-` is literal.
        assert!(glob_matches("]", "[]]"));
        assert!(glob_matches("-", "[-a]"));
        assert!(glob_matches("-", "[a-]"));
        assert!(glob_matches("a", "[a-]"));
        // `!` is NOT a negation character in SQLite GLOB — it is a literal member.
        assert!(glob_matches("!", "[!a]"));
        assert!(glob_matches("a", "[!a]"));
        assert!(!glob_matches("b", "[!a]"));
        // An unterminated `[` is a literal `[`.
        assert!(glob_matches("[x", "[x"));
        assert!(!glob_matches("ax", "[x"));
    }

    #[test]
    fn matching_terminates_on_adversarial_patterns() {
        // A pattern full of wildcards against a long non-matching string must not
        // blow up — the bounded algorithm returns quickly.
        let text: String = "a".repeat(64);
        assert!(!like_matches(&text, &format!("{}b", "%".repeat(32)), None));
        assert!(like_matches(&text, &"%".repeat(40), None));
        let gtext: String = "a".repeat(64);
        assert!(!glob_matches(&gtext, &format!("{}b", "*".repeat(32))));
    }

    #[test]
    fn multibyte_single_char_wildcards() {
        // `_`/`?` match one Unicode character, not one byte.
        assert!(like_matches("é", "_", None));
        assert!(glob_matches("é", "?"));
        assert!(like_matches("café", "ca__", None));
    }
}
