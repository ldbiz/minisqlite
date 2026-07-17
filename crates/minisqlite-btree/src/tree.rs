//! Database and table-root construction, plus the one write primitive the whole
//! b-tree routes through (`put_page`), which preserves page 1's database header.

use minisqlite_fileformat::{DatabaseHeader, PageBuilder, PageType, HEADER_SIZE};
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

/// Offset of the one-byte "bytes of reserved space at the end of each page" field
/// in the 100-byte database header on page 1 (fileformat2 §1.3; the same field
/// `DatabaseHeader::reserved_space` decodes). `usable_of` reads just this byte so
/// it need not decode — and stay in sync with the validity of — the whole header.
const RESERVED_SPACE_OFFSET: usize = 20;

/// Smallest usable page size the format permits (fileformat2 §1.3.4). A reserved
/// count that would drop `U` below this is corrupt rather than a real reserved
/// region, so `usable_of` ignores it and treats the page as fully usable.
const MIN_USABLE_SIZE: usize = 480;

/// Usable bytes per page (`U`): the page size minus the reserved region at the end
/// of every page (fileformat2 §1.3.4). This is the single source of `U` for the
/// whole b-tree — every cell-layout and overflow-spill computation sizes against it
/// — so honoring a file's reserved region here corrects `U` everywhere at once.
///
/// The reserved count is the byte at `RESERVED_SPACE_OFFSET` of page 1's database
/// header, read fresh on each call (page 1 is the hottest page, so this is cheap).
/// `U` falls back to the full `page_size` (reserved treated as 0) in the three
/// cases where there is no trustworthy reserved byte:
///   * page 1 does not exist yet — `init_database` calls this on an empty pager
///     *before* it allocates page 1, so the read errors and must not propagate;
///   * page 1 is too short to contain the byte at `RESERVED_SPACE_OFFSET`;
///   * the reserved byte would push `U` below `MIN_USABLE_SIZE`, i.e. it is corrupt.
///
/// A freshly created database reserves nothing, so `U == page_size`; a file real
/// sqlite wrote with a non-zero reserved region shrinks `U` to match.
pub(crate) fn usable_of(pager: &dyn Pager) -> usize {
    let page_size = pager.page_size() as usize;
    let reserved = pager
        .read_page(1)
        .ok()
        .and_then(|page| page.get(RESERVED_SPACE_OFFSET).copied())
        .unwrap_or(0) as usize;
    let usable = page_size.saturating_sub(reserved);
    if usable < MIN_USABLE_SIZE { page_size } else { usable }
}

/// Write a finished page, preserving page 1's 100-byte database header.
///
/// A page rebuilt through `PageBuilder` for page 1 leaves its first 100 bytes
/// zeroed (the builder only owns the b-tree region at offset 100). Page 1 always
/// carries the database header there, so we overlay the current header bytes back
/// before handing the page to the pager. For every other page this is a plain
/// write.
pub(crate) fn put_page(pager: &mut dyn Pager, id: PageId, mut bytes: Vec<u8>) -> Result<()> {
    if id == 1 {
        let header: [u8; HEADER_SIZE] = pager
            .read_page(1)?
            .get(0..HEADER_SIZE)
            .ok_or_else(|| Error::format("page 1 is shorter than the database header"))?
            .try_into()
            .expect("a slice of length HEADER_SIZE converts to [u8; HEADER_SIZE]");
        bytes[0..HEADER_SIZE].copy_from_slice(&header);
    }
    pager.write_page(id, &bytes)
}

/// Format a fresh, empty database: allocate page 1 and initialize it as the empty
/// `sqlite_schema` table b-tree (a leaf table page) with the 100-byte database
/// header. Page 1 is always the root of `sqlite_schema`.
///
/// Called once when creating a new (in-memory or on-disk) database, on an empty
/// pager. `allocate_page` returns 1 on an empty pager; anything else means the
/// pager was not fresh, which is a caller error we surface rather than corrupt.
pub fn init_database(pager: &mut dyn Pager) -> Result<()> {
    let page_size = pager.page_size();
    let usable = usable_of(pager);
    let id = pager.allocate_page()?;
    if id != 1 {
        return Err(Error::format(format!(
            "init_database expects an empty pager; allocate_page returned {id}, not 1"
        )));
    }
    let mut page = PageBuilder::new(page_size as usize, usable, 1, PageType::LeafTable).finish();
    // A fresh header for this page size. `database_size_pages` is 1 now and is left
    // for the pager/on-disk layer to maintain as pages are allocated — the in-memory
    // page count is authoritative until then.
    let header = DatabaseHeader { page_size, ..DatabaseHeader::default() };
    page[0..HEADER_SIZE].copy_from_slice(&header.to_bytes());
    pager.write_page(1, &page)
}

/// Create a new, empty table b-tree and return its root page id (>= 2). Formats a
/// freshly allocated page as an empty leaf table page; the id is what
/// `TableDef.root_page` stores. Must be called after `init_database` (page 1 is
/// already taken by `sqlite_schema`).
pub fn create_table_btree(pager: &mut dyn Pager) -> Result<PageId> {
    let page_size = pager.page_size() as usize;
    let usable = usable_of(pager);
    let id = pager.allocate_page()?;
    let page = PageBuilder::new(page_size, usable, id, PageType::LeafTable).finish();
    pager.write_page(id, &page)?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_fileformat::{DatabaseHeader, PageView};
    use minisqlite_pager::MemPager;

    #[test]
    fn init_database_makes_page1_schema_leaf_with_header() {
        let mut p = MemPager::new(4096);
        init_database(&mut p).unwrap();
        assert_eq!(p.page_count().unwrap(), 1);

        let page = p.read_page(1).unwrap();
        // The database header sits in the first 100 bytes and round-trips.
        let hdr_bytes: [u8; HEADER_SIZE] = page[0..HEADER_SIZE].try_into().unwrap();
        let hdr = DatabaseHeader::read(&hdr_bytes).unwrap();
        assert_eq!(hdr.page_size, 4096);
        // The b-tree header at offset 100 is an empty leaf table page.
        let view = PageView::new(page, 1, 4096).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable);
        assert_eq!(view.cell_count(), 0);
    }

    #[test]
    fn init_database_carries_page_size_into_header() {
        for ps in [512u32, 4096, 65536] {
            let mut p = MemPager::new(ps);
            init_database(&mut p).unwrap();
            let page = p.read_page(1).unwrap();
            let hdr_bytes: [u8; HEADER_SIZE] = page[0..HEADER_SIZE].try_into().unwrap();
            assert_eq!(DatabaseHeader::read(&hdr_bytes).unwrap().page_size, ps);
        }
    }

    #[test]
    fn create_table_btree_allocates_empty_leaf_roots() {
        let mut p = MemPager::new(4096);
        init_database(&mut p).unwrap();
        let a = create_table_btree(&mut p).unwrap();
        let b = create_table_btree(&mut p).unwrap();
        assert_eq!((a, b), (2, 3));
        for id in [a, b] {
            let view = PageView::new(p.read_page(id).unwrap(), id, 4096).unwrap();
            assert_eq!(view.page_type(), PageType::LeafTable);
            assert_eq!(view.cell_count(), 0);
        }
    }

    #[test]
    fn init_database_rejects_non_empty_pager() {
        let mut p = MemPager::new(4096);
        p.allocate_page().unwrap(); // now page 1 exists; init should refuse
        assert!(init_database(&mut p).is_err());
    }

    #[test]
    fn put_page_preserves_page1_header() {
        let mut p = MemPager::new(4096);
        init_database(&mut p).unwrap();
        let before: [u8; HEADER_SIZE] = p.read_page(1).unwrap()[0..HEADER_SIZE].try_into().unwrap();
        // Rebuild page 1 as an interior page (zeroing 0..100 in the builder output)
        // and write it through put_page; the header must survive.
        let rebuilt = PageBuilder::new(4096, 4096, 1, PageType::InteriorTable).finish();
        assert_eq!(&rebuilt[0..HEADER_SIZE], &[0u8; HEADER_SIZE]); // builder zeroed it
        put_page(&mut p, 1, rebuilt).unwrap();
        let after: [u8; HEADER_SIZE] = p.read_page(1).unwrap()[0..HEADER_SIZE].try_into().unwrap();
        assert_eq!(before, after, "page 1 database header must be preserved by put_page");
    }

    /// Overwrite page 1's reserved-space byte (offset 20) and write the page back,
    /// simulating a `.db` file real sqlite wrote with a `reserved`-byte tail region.
    /// MemPager writes outside a transaction apply directly, so the change sticks.
    fn set_reserved(p: &mut MemPager, reserved: u8) {
        let mut page1 = p.read_page(1).unwrap().to_vec();
        page1[RESERVED_SPACE_OFFSET] = reserved;
        p.write_page(1, &page1).unwrap();
    }

    #[test]
    fn usable_of_is_full_page_size_for_fresh_database() {
        // A freshly created database reserves nothing (byte 20 == 0), so the usable
        // size is the whole page — for every legal page size, not just the default.
        for ps in [512u32, 4096, 65536] {
            let mut p = MemPager::new(ps);
            init_database(&mut p).unwrap();
            assert_eq!(usable_of(&p), ps as usize, "fresh db reserves nothing (page {ps})");
        }
    }

    #[test]
    fn usable_of_subtracts_reserved_region() {
        let mut p = MemPager::new(4096);
        init_database(&mut p).unwrap();
        set_reserved(&mut p, 32);
        assert_eq!(usable_of(&p), 4096 - 32, "U = page_size - reserved");
    }

    #[test]
    fn usable_of_reads_reserved_from_documented_offset_20() {
        // Independently pin the file-format contract that the reserved-space field
        // lives at header offset 20 (fileformat2 §1.3). This writes via a LITERAL 20
        // rather than `RESERVED_SPACE_OFFSET`, so a drift of the const to a byte that
        // is zero in a fresh header (e.g. 24, the file_change_counter) is caught here
        // instead of moving symmetrically with `set_reserved` and slipping through.
        let mut p = MemPager::new(4096);
        init_database(&mut p).unwrap();
        let mut page1 = p.read_page(1).unwrap().to_vec();
        page1[20] = 24;
        p.write_page(1, &page1).unwrap();
        assert_eq!(usable_of(&p), 4096 - 24, "reserved byte is read from offset 20");
        assert_eq!(RESERVED_SPACE_OFFSET, 20, "reserved-space field is at header offset 20");
    }

    #[test]
    fn usable_of_defaults_to_page_size_without_page1() {
        // Exactly the state `init_database` calls `usable_of` in: page 1 not yet
        // allocated. It must return the full page size and never panic.
        let p = MemPager::new(4096);
        assert!(p.read_page(1).is_err(), "precondition: page 1 is absent");
        assert_eq!(usable_of(&p), 4096);
    }

    #[test]
    fn usable_of_ignores_reserved_below_min_usable() {
        // On a 512-byte page, reserved 33 => usable 479 < 480: corrupt, so U falls
        // back to the full page. Reserved 32 => usable 480, exactly the floor, kept.
        let mut p = MemPager::new(512);
        init_database(&mut p).unwrap();
        set_reserved(&mut p, 33);
        assert_eq!(usable_of(&p), 512, "reserved below the 480 floor is ignored");
        set_reserved(&mut p, 32);
        assert_eq!(usable_of(&p), 480, "reserved at exactly the 480 floor is honored");
    }

    #[test]
    fn usable_of_defaults_to_page_size_for_short_page1() {
        // A page 1 shorter than `RESERVED_SPACE_OFFSET + 1` has no reserved byte to
        // read, so `usable_of` must default to the full page. MemPager only ever
        // returns full-size pages, so a tiny stub pager exercises this guard.
        struct ShortPager {
            page: [u8; RESERVED_SPACE_OFFSET],
        }
        impl Pager for ShortPager {
            fn read_page(&self, id: PageId) -> Result<&[u8]> {
                assert_eq!(id, 1, "usable_of reads only page 1");
                Ok(&self.page)
            }
            fn page_count(&self) -> Result<PageId> {
                Ok(1)
            }
            fn page_size(&self) -> u32 {
                4096
            }
            fn begin(&mut self) -> Result<()> {
                unreachable!("usable_of does not begin a transaction")
            }
            fn write_page(&mut self, _id: PageId, _bytes: &[u8]) -> Result<()> {
                unreachable!("usable_of does not write")
            }
            fn allocate_page(&mut self) -> Result<PageId> {
                unreachable!("usable_of does not allocate")
            }
            fn free_page(&mut self, _id: PageId) -> Result<()> {
                unreachable!("usable_of does not free")
            }
            fn commit(&mut self) -> Result<()> {
                unreachable!("usable_of does not commit")
            }
            fn rollback(&mut self) -> Result<()> {
                unreachable!("usable_of does not roll back")
            }
        }
        let p = ShortPager { page: [0u8; RESERVED_SPACE_OFFSET] };
        assert_eq!(usable_of(&p), 4096);
    }
}
