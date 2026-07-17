//! Seam guard — the engine route is a fixed set of one-per-crate traits and named
//! entrypoints, and the workspace stays a single live path. This is an
//! architectural invariant, not a correctness check: correctness is checked
//! elsewhere; this keeps the route singular and the tree clean so the pinned facade
//! can be expand-then-contract behind the seams and drift is a build failure rather
//! than a hand-maintained "which path is active" contract.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The owned seam traits: each must be declared exactly once, in its named crate.
const SEAM_TRAITS: &[(&str, &str)] = &[
    ("Engine", "minisqlite-engine"),
    ("Pager", "minisqlite-pager"),
    ("Catalog", "minisqlite-catalog"),
    ("Planner", "minisqlite-plan"),
    ("Executor", "minisqlite-exec"),
];

/// The facade crate an external caller links. It has no in-workspace reverse dependency by
/// design (it is linked from outside the workspace), so it is exempt from the
/// no-orphan-crate rule.
const FACADE_CRATE: &str = "minisqlite";

/// Backup / scratch file suffixes that must never appear in the tree; the workspace
/// holds source only, with no parked copies.
const FORBIDDEN_SUFFIXES: &[&str] =
    &[".bak", ".orig", ".tmp", ".rej", ".swp", ".DELETE", ".canonical", ".flat_survivor"];

/// Walk up from this crate's manifest dir to the workspace root (the Cargo.toml
/// that declares `[workspace]`).
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists()
            && fs::read_to_string(&manifest).map(|s| s.contains("[workspace]")).unwrap_or(false)
        {
            return dir;
        }
        if !dir.pop() {
            panic!("workspace root (a Cargo.toml with [workspace]) not found above CARGO_MANIFEST_DIR");
        }
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip build output and hidden dirs (e.g. target/, .git/).
            if name == "target" || name.starts_with('.') {
                continue;
            }
            collect_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn rust_files(crates_dir: &Path) -> Vec<PathBuf> {
    let mut all = Vec::new();
    collect_files(crates_dir, &mut all);
    all.into_iter().filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs")).collect()
}

fn crate_name_of(file: &Path, crates_dir: &Path) -> Option<String> {
    file.strip_prefix(crates_dir).ok()?.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned())
}

fn crate_dirs(crates_dir: &Path) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(crates_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(n) = entry.file_name().to_str() {
                    set.insert(n.to_string());
                }
            }
        }
    }
    set
}

/// True if `line` declares `pub trait <name>` and the next char ends the
/// identifier, so `Engine` does not match `EngineExt`.
fn declares_pub_trait(line: &str, name: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix(&format!("pub trait {name}")) else {
        return false;
    };
    matches!(rest.chars().next(), None | Some(' ') | Some('<') | Some('{') | Some(':'))
}

/// True if `line` declares `pub fn <name>` (function, not a `pub use` re-export).
fn declares_pub_fn(line: &str, name: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix(&format!("pub fn {name}")) else {
        return false;
    };
    matches!(rest.chars().next(), Some('(') | Some('<') | Some(' '))
}

/// Each owned seam trait is declared exactly once, in its named crate. A second
/// declaration (anywhere) or a declaration in the wrong crate is a forked route.
#[test]
fn one_trait_per_named_seam_in_its_crate() {
    let crates_dir = workspace_root().join("crates");
    let files = rust_files(&crates_dir);

    for (trait_name, owner) in SEAM_TRAITS {
        let mut decl_crates: Vec<String> = Vec::new();
        for file in &files {
            let Ok(text) = fs::read_to_string(file) else {
                continue;
            };
            let count = text.lines().filter(|l| declares_pub_trait(l, trait_name)).count();
            for _ in 0..count {
                if let Some(name) = crate_name_of(file, &crates_dir) {
                    decl_crates.push(name);
                }
            }
        }
        assert_eq!(
            decl_crates.len(),
            1,
            "SEAM INVARIANT: `pub trait {trait_name}` must be declared exactly once; \
             found {} declaration(s) in {decl_crates:?}. The seam stays a single trait so the route cannot fork.",
            decl_crates.len()
        );
        assert_eq!(
            &decl_crates[0], owner,
            "SEAM INVARIANT: `pub trait {trait_name}` must live in `{owner}`; found in `{}`.",
            decl_crates[0]
        );
    }
}

/// The SQL front end has exactly one `pub fn parse` entrypoint, in `minisqlite-sql`.
#[test]
fn one_parse_entrypoint_in_minisqlite_sql() {
    let crates_dir = workspace_root().join("crates");
    let files = rust_files(&crates_dir);
    let mut decl_crates: Vec<String> = Vec::new();
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        let count = text.lines().filter(|l| declares_pub_fn(l, "parse")).count();
        for _ in 0..count {
            if let Some(name) = crate_name_of(file, &crates_dir) {
                decl_crates.push(name);
            }
        }
    }
    assert_eq!(
        decl_crates.len(),
        1,
        "SEAM INVARIANT: the SQL front end must have exactly one `pub fn parse` entrypoint; \
         found {} in {decl_crates:?}.",
        decl_crates.len()
    );
    assert_eq!(
        decl_crates[0], "minisqlite-sql",
        "SEAM INVARIANT: `pub fn parse` must live in `minisqlite-sql`; found in `{}`.",
        decl_crates[0]
    );
}

/// Every component crate is depended on by at least one other crate. An orphan
/// (zero reverse dependencies) is a crate the route no longer uses — an
/// abandonment/fork signal. The facade is the one exempt entrypoint (linked from
/// outside the workspace by an external caller).
#[test]
fn no_orphan_crate() {
    let crates_dir = workspace_root().join("crates");
    let dirs = crate_dirs(&crates_dir);

    let mut referenced: BTreeSet<String> = BTreeSet::new();
    for owner in &dirs {
        let manifest = crates_dir.join(owner).join("Cargo.toml");
        let text = fs::read_to_string(&manifest).unwrap_or_default();
        for other in &dirs {
            if other == owner {
                continue;
            }
            if text.contains(&format!("\"../{other}\"")) {
                referenced.insert(other.clone());
            }
        }
    }

    let orphans: Vec<&String> =
        dirs.iter().filter(|c| c.as_str() != FACADE_CRATE && !referenced.contains(*c)).collect();
    assert!(
        orphans.is_empty(),
        "SEAM INVARIANT: orphan crate(s) with no reverse dependency: {orphans:?}. \
         Every component crate must be depended on by another; `{FACADE_CRATE}` is the only exempt entrypoint. \
         Wire it into the route or remove it — do not park a competing implementation."
    );
}

/// No crate selects behavior with cargo features. There is one build and one live
/// path; variants are split into crates and chosen by code, never feature-gated.
#[test]
fn no_behavior_selecting_cargo_features() {
    let crates_dir = workspace_root().join("crates");
    let mut offenders: Vec<String> = Vec::new();
    for c in crate_dirs(&crates_dir) {
        let text = fs::read_to_string(crates_dir.join(&c).join("Cargo.toml")).unwrap_or_default();
        let mut in_features = false;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                in_features = t == "[features]";
                continue;
            }
            if in_features {
                let code = t.split('#').next().unwrap_or("").trim();
                if !code.is_empty() && code.contains('=') {
                    offenders.push(c.clone());
                    break;
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "SEAM INVARIANT: crate(s) declare a non-empty [features] table: {offenders:?}. \
         Behavior must not be selected by cargo features — there is one build and one live path."
    );
}

/// Source files are valid Rust only (no injected non-code markers), and the tree
/// carries no backup/scratch files.
#[test]
fn no_injected_markers_or_backup_files() {
    let crates_dir = workspace_root().join("crates");
    let mut all = Vec::new();
    collect_files(&crates_dir, &mut all);

    let mut backups: Vec<String> = Vec::new();
    for file in &all {
        let name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let is_backup =
            name.ends_with('~') || FORBIDDEN_SUFFIXES.iter().any(|suffix| name.ends_with(suffix));
        if is_backup {
            backups.push(file.strip_prefix(&crates_dir).unwrap_or(file).display().to_string());
        }
    }
    assert!(
        backups.is_empty(),
        "SEAM INVARIANT: backup/scratch file(s) in the tree: {backups:?}. The workspace holds source only."
    );

    // The HTML comment opener is never valid Rust, so an injected HTML comment in a
    // `.rs` file is corruption, not code. Only the opener is flagged (the closer is
    // an ordinary prose arrow in comments). Built via concat! so this guard never
    // contains -- and therefore never flags -- the marker itself.
    let html_open = concat!("<!", "--");
    let mut marked: Vec<String> = Vec::new();
    for file in all.iter().filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs")) {
        let text = fs::read_to_string(file).unwrap_or_default();
        if text.contains(html_open) {
            marked.push(file.strip_prefix(&crates_dir).unwrap_or(file).display().to_string());
        }
    }
    assert!(
        marked.is_empty(),
        "SEAM INVARIANT: Rust source with injected non-code markers (HTML comments): {marked:?}. \
         Source files are valid Rust only."
    );
}
