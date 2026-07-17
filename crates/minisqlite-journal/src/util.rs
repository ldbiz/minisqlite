//! Small internal helpers shared by the codec, writer, and recovery paths:
//! big-endian 4-byte accessors (the journal, like the rest of the file format, is
//! big-endian) and the directory-entry-durability helpers (`fsync_parent_dir` and
//! `remove_file_durable`) that make a journal's creation and removal survive a crash.

use std::fs::File;
use std::path::Path;

use minisqlite_types::Result;

/// Read a big-endian `u32` at `off`. The caller guarantees `buf` holds four bytes
/// at `off` (fixed-size header/record buffers), so an out-of-range offset is a bug,
/// not on-disk corruption — it panics rather than silently returning zero.
#[inline]
pub(crate) fn be32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Write a big-endian `u32` at `off`. Same bounds contract as [`be32`].
#[inline]
pub(crate) fn write_be32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
}

/// Round `v` up to the next multiple of `align`. `align` is a validated sector
/// size (a nonzero power of two) at every call site; the guard keeps a zero from
/// dividing by zero if that ever changes.
#[inline]
pub(crate) fn round_up(v: u64, align: u64) -> u64 {
    if align == 0 {
        return v;
    }
    v.div_ceil(align) * align
}

/// fsync the directory containing `path`, returning any genuine sync error.
///
/// On Unix a file's creation, rename, or unlink is only durable once the *directory*
/// entry is flushed: `fsync` on the file flushes its data+inode but NOT the containing
/// directory entry, so any operation that adds or removes a name in the directory
/// (creating a journal, unlinking it) must pair the file fsync with this.
///
/// Opening a directory to fsync it is not portable, so if the directory cannot be
/// opened at all we treat the sync as unavailable (`Ok`) rather than an error — the
/// file's own fsync is the primary durability step. A directory that DOES open but
/// fails to sync is a genuine failure and is returned, so callers decide what to do
/// with it: the create side (where the journal's *name* is itself load-bearing — the
/// pager must not touch the db until the name is durable) propagates it via `?` and
/// fails closed; the unlink side (belt-and-suspenders, since the file's own fsync
/// already ran and recovery is idempotent) ignores it.
pub(crate) fn fsync_parent_dir(path: &Path) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let dir = if parent.as_os_str().is_empty() { Path::new(".") } else { parent };
    match File::open(dir) {
        Ok(f) => Ok(f.sync_all()?),
        Err(_) => Ok(()),
    }
}

/// Durably remove `path`: unlink it, then fsync the parent directory so the removal
/// survives a power loss (see [`fsync_parent_dir`]). A file that is already gone is
/// treated as success — the desired post-state (the name does not exist) already
/// holds, which is exactly what an interrupted-and-retried removal needs. This is the
/// one place the durable-unlink protocol lives; both the DELETE-mode commit and the
/// end of recovery call it, so a future change to how the journal is retired touches
/// a single site.
///
/// The dir fsync here is best-effort (its error is swallowed): the unlink itself is
/// the load-bearing step, and if a crash drops the not-yet-durable removal the journal
/// simply reappears and recovery re-runs its idempotent rollback — no data is lost.
pub(crate) fn remove_file_durable(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let _ = fsync_parent_dir(path);
    Ok(())
}
