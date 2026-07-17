//! The `sqlite_schema` row: the five-column record every schema object is stored as
//! on page 1 (fileformat2 §2.6, `schematab.html`). Columns, in order:
//!   `type` TEXT · `name` TEXT · `tbl_name` TEXT · `rootpage` INTEGER · `sql` TEXT.
//!
//! This is the on-disk shape of one schema entry; the higher-level `def` module's
//! structs are what the rest of the engine reads. `SchemaRow` is the narrow seam
//! between the two: [`SchemaCatalog`](crate::SchemaCatalog) encodes a `SchemaRow`
//! to persist a `CREATE TABLE` and decodes one per row when it rebuilds the cache.

use minisqlite_fileformat::{decode_record, encode_record, encode_record_enc, TextEncoding};
use minisqlite_types::{Error, Result, Value};

/// One decoded `sqlite_schema` row.
///
/// `rootpage` is the b-tree root page number (0 for views/triggers, which have no
/// b-tree). `sql` is the verbatim `CREATE` statement text, or `None` where the
/// format stores NULL (auto-created indexes; views/triggers still carry text).
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaRow {
    pub obj_type: String,
    pub name: String,
    pub tbl_name: String,
    pub rootpage: i64,
    pub sql: Option<String>,
}

impl SchemaRow {
    /// The five column values, in schema order, exactly as stored: the three text
    /// columns as `Text`, `rootpage` as `Integer`, and `sql` as `Text` or `Null`.
    pub fn to_values(&self) -> Vec<Value> {
        vec![
            Value::Text(self.obj_type.clone()),
            Value::Text(self.name.clone()),
            Value::Text(self.tbl_name.clone()),
            Value::Integer(self.rootpage),
            match &self.sql {
                Some(s) => Value::Text(s.clone()),
                None => Value::Null,
            },
        ]
    }

    /// Decode a `SchemaRow` from a record's values. `type` and `name` are required
    /// text. `tbl_name` must be text when present, but a record too short to include
    /// it defaults to "" (SQLite short-record semantics); a *present* `Null` or
    /// non-text `tbl_name` fails closed — a full-length `sqlite_schema` row always
    /// carries the object/table name here, so an explicit NULL is corruption, not a
    /// default. `rootpage` maps `Null`/absent to 0; `sql` maps `Null`/absent to
    /// `None`. A record with fewer than five values treats the missing trailing
    /// columns as NULL. Any present column that is not its expected type fails closed
    /// (`Error::Format`) rather than silently coercing a corrupt schema.
    pub fn from_values(vals: &[Value]) -> Result<SchemaRow> {
        let obj_type = required_text(vals, 0, "type")?;
        let name = required_text(vals, 1, "name")?;
        let tbl_name = text_or_absent(vals, 2, "tbl_name")?;
        let rootpage = match vals.get(3) {
            None | Some(Value::Null) => 0,
            Some(Value::Integer(i)) => *i,
            Some(other) => {
                return Err(Error::Format(format!(
                    "sqlite_schema rootpage column is not an integer: {other:?}"
                )));
            }
        };
        let sql = match vals.get(4) {
            None | Some(Value::Null) => None,
            Some(Value::Text(s)) => Some(s.clone()),
            Some(other) => {
                return Err(Error::Format(format!(
                    "sqlite_schema sql column is not text: {other:?}"
                )));
            }
        };
        Ok(SchemaRow { obj_type, name, tbl_name, rootpage, sql })
    }

    /// Encode this row to its on-disk record bytes (TEXT in UTF-8).
    pub fn to_record(&self) -> Vec<u8> {
        encode_record(&self.to_values())
    }

    /// Encode this row to its on-disk record bytes with TEXT stored in encoding
    /// `enc` (§1.3.13). A UTF-16 database's `sqlite_schema` rows are themselves
    /// UTF-16, so the object names and `CREATE` text this persists must be laid down
    /// in the database's encoding — otherwise real sqlite would read the schema back
    /// as garbage. With `enc == Utf8` this is identical to [`to_record`](Self::to_record).
    pub fn to_record_enc(&self, enc: TextEncoding) -> Vec<u8> {
        encode_record_enc(&self.to_values(), enc)
    }

    /// Decode a row from its on-disk record bytes.
    pub fn from_record(buf: &[u8]) -> Result<SchemaRow> {
        Self::from_values(&decode_record(buf))
    }
}

/// A required text column: present and `Text`, else a format error.
fn required_text(vals: &[Value], idx: usize, col: &str) -> Result<String> {
    match vals.get(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => {
            Err(Error::Format(format!("sqlite_schema {col} column is not text: {other:?}")))
        }
        None => Err(Error::Format(format!("sqlite_schema record missing required {col} column"))),
    }
}

/// A text column that must be `Text` when present, but whose absence in a short
/// record is tolerated as an empty string (SQLite short-record semantics). A
/// present `Null` fails closed like any other non-text value: a full-length
/// `sqlite_schema` row never stores NULL here, so an explicit NULL is corruption,
/// distinct from a trailing column simply omitted.
fn text_or_absent(vals: &[Value], idx: usize, col: &str) -> Result<String> {
    match vals.get(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        None => Ok(String::new()),
        Some(other) => {
            Err(Error::Format(format!("sqlite_schema {col} column is not text: {other:?}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_values_has_exact_shape() {
        // Distinct name vs tbl_name (an index row) so a name<->tbl_name transposition
        // in `to_values` is caught — a table row (name == tbl_name) would mask it.
        let row = SchemaRow {
            obj_type: "index".into(),
            name: "ix_name".into(),
            tbl_name: "widgets".into(),
            rootpage: 7,
            sql: Some("CREATE INDEX ix_name ON widgets(c)".into()),
        };
        let vals = row.to_values();
        assert_eq!(vals.len(), 5);
        assert!(matches!(&vals[0], Value::Text(s) if s == "index"));
        assert!(matches!(&vals[1], Value::Text(s) if s == "ix_name"));
        assert!(matches!(&vals[2], Value::Text(s) if s == "widgets"));
        assert!(matches!(&vals[3], Value::Integer(7)));
        assert!(matches!(&vals[4], Value::Text(s) if s == "CREATE INDEX ix_name ON widgets(c)"));
    }

    #[test]
    fn from_values_maps_columns_by_position() {
        // Distinct value per column so a decoder that transposed positions (e.g.
        // name<->tbl_name) is caught rather than hidden by a table row's name == tbl_name.
        let vals = vec![
            Value::Text("index".into()),
            Value::Text("ix_name".into()),
            Value::Text("widgets".into()),
            Value::Integer(9),
            Value::Text("CREATE INDEX ix_name ON widgets(c)".into()),
        ];
        let row = SchemaRow::from_values(&vals).unwrap();
        assert_eq!(row.obj_type, "index");
        assert_eq!(row.name, "ix_name");
        assert_eq!(row.tbl_name, "widgets");
        assert_eq!(row.rootpage, 9);
        assert_eq!(row.sql.as_deref(), Some("CREATE INDEX ix_name ON widgets(c)"));
    }

    #[test]
    fn table_row_record_roundtrips() {
        let row = SchemaRow {
            obj_type: "table".into(),
            name: "widgets".into(),
            tbl_name: "widgets".into(),
            rootpage: 4,
            sql: Some("CREATE TABLE widgets(id INTEGER PRIMARY KEY, name TEXT)".into()),
        };
        let back = SchemaRow::from_record(&row.to_record()).unwrap();
        assert_eq!(back, row);
    }

    #[test]
    fn view_row_record_roundtrips_with_zero_root_and_null_sql() {
        // A view has no b-tree (rootpage 0). A NULL sql column must round-trip to
        // None (auto-index shape), so the codec cannot conflate it with "".
        let row = SchemaRow {
            obj_type: "view".into(),
            name: "v".into(),
            tbl_name: "v".into(),
            rootpage: 0,
            sql: None,
        };
        let back = SchemaRow::from_record(&row.to_record()).unwrap();
        assert_eq!(back, row);
        assert_eq!(back.rootpage, 0);
        assert!(back.sql.is_none());
    }

    #[test]
    fn short_record_treats_missing_trailing_columns_as_null() {
        // Only type + name present: tbl_name -> "", rootpage -> 0, sql -> None.
        let vals = vec![Value::Text("table".into()), Value::Text("t".into())];
        let row = SchemaRow::from_values(&vals).unwrap();
        assert_eq!(row.obj_type, "table");
        assert_eq!(row.name, "t");
        assert_eq!(row.tbl_name, "");
        assert_eq!(row.rootpage, 0);
        assert!(row.sql.is_none());
    }

    #[test]
    fn present_null_tbl_name_fails_closed() {
        // A full-length row with an explicit NULL tbl_name is corrupt (real sqlite
        // always writes the name here). This is distinct from the short record above,
        // which omits the column entirely and is tolerated.
        let vals =
            vec![Value::Text("table".into()), Value::Text("t".into()), Value::Null];
        assert!(
            SchemaRow::from_values(&vals).is_err(),
            "an explicit NULL tbl_name must fail closed, not default to \"\""
        );
    }

    #[test]
    fn missing_required_columns_fail_closed() {
        // Fewer than two values, or a non-text type/name, is corrupt.
        assert!(SchemaRow::from_values(&[]).is_err());
        assert!(SchemaRow::from_values(&[Value::Text("table".into())]).is_err());
        assert!(
            SchemaRow::from_values(&[Value::Integer(1), Value::Text("t".into())]).is_err(),
            "non-text type column must fail closed"
        );
    }

    #[test]
    fn wrong_typed_optional_columns_fail_closed() {
        // rootpage present but not an integer, and sql present but not text, are
        // corrupt schemas — surfaced rather than coerced.
        let bad_root = vec![
            Value::Text("table".into()),
            Value::Text("t".into()),
            Value::Text("t".into()),
            Value::Text("not a number".into()),
        ];
        assert!(SchemaRow::from_values(&bad_root).is_err());

        let bad_sql = vec![
            Value::Text("table".into()),
            Value::Text("t".into()),
            Value::Text("t".into()),
            Value::Integer(2),
            Value::Integer(5),
        ];
        assert!(SchemaRow::from_values(&bad_sql).is_err());
    }
}
