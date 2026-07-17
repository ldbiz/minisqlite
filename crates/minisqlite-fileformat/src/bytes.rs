//! Big-endian integer accessors shared by the header and page codecs. All
//! multibyte integers in the SQLite file format are big-endian, so these are the
//! one place that byte order is spelled out. The `try_*` forms fail closed on a
//! short slice (used where an offset comes from on-disk data and could be corrupt);
//! the panicking forms are for fixed-size buffers whose bounds are guaranteed by
//! the caller (the 100-byte header, an in-bounds page-header field).

#[inline]
pub(crate) fn be16(buf: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([buf[off], buf[off + 1]])
}

#[inline]
pub(crate) fn be32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
pub(crate) fn try_be32(buf: &[u8], off: usize) -> Option<u32> {
    let b = buf.get(off..off + 4)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline]
pub(crate) fn write_be16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_be_bytes());
}

#[inline]
pub(crate) fn write_be32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
}
