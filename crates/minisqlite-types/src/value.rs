//! The dynamically-typed SQLite value (the five storage classes) plus the row
//! and result-set aggregates. This is the vocabulary every other crate speaks;
//! affinity, coercion, comparison, and CAST all operate on `Value` and live in
//! sibling modules (`affinity`, `numeric`, `compare`, `cast`).

/// A dynamically-typed SQLite value (the five storage classes). Affinity and
/// coercion happen during evaluation; a stored/returned value is one of these.
///
/// The variant set is stable: the facade `minisqlite` re-exports `Value`, so
/// these five variants must not be renamed or removed.
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    /// The name SQLite's `typeof()` reports for this value's *stored* storage
    /// class: one of `"null"`, `"integer"`, `"real"`, `"text"`, `"blob"`.
    /// Affinity mistakes surface directly through `typeof()`, so this must report
    /// the concrete variant, never an affinity or a "preferred" type.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Integer(_) => "integer",
            Value::Real(_) => "real",
            Value::Text(_) => "text",
            Value::Blob(_) => "blob",
        }
    }

    /// True for the NULL storage class. Convenience for the many call sites that
    /// branch on NULL before applying three-valued logic.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// True for the numeric storage classes (INTEGER or REAL). SQLite treats the
    /// two as one "numeric" class for comparison and grouping, so this is the
    /// predicate that class-based logic keys on.
    pub fn is_numeric(&self) -> bool {
        matches!(self, Value::Integer(_) | Value::Real(_))
    }
}

/// One result row: one `Value` per column, in `SELECT` order.
pub type Row = Vec<Value>;

/// A query's result set: the column names (in `SELECT` order) and the rows. The
/// names matter — clients like the TCL `db eval {sql} {body}` form bind a variable
/// per column name, so a real engine must report them.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Row>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_name_reports_stored_class() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Integer(1).type_name(), "integer");
        assert_eq!(Value::Real(1.0).type_name(), "real");
        assert_eq!(Value::Text("x".into()).type_name(), "text");
        assert_eq!(Value::Blob(vec![0]).type_name(), "blob");
    }

    #[test]
    fn null_and_numeric_predicates() {
        assert!(Value::Null.is_null());
        assert!(!Value::Integer(0).is_null());
        assert!(Value::Integer(0).is_numeric());
        assert!(Value::Real(0.0).is_numeric());
        assert!(!Value::Text("1".into()).is_numeric());
        assert!(!Value::Null.is_numeric());
    }
}
