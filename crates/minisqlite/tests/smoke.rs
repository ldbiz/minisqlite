//! Smoke set — a tiny provided battery; the measurement harness also uses it. Basic
//! SQL the engine must get right: these fail until the engine exists, which is
//! expected, not a gate. Do not edit them to pass — implement the engine. Broader
//! coverage comes from a far larger differential vs real sqlite3.

use minisqlite::{Connection, Value};

fn one(db: &mut Connection, sql: &str) -> Vec<Vec<Value>> {
    db.query(sql).expect("query should succeed").rows
}

#[test]
fn create_insert_select_ordered() {
    let mut db = Connection::open_in_memory().unwrap();
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();
    db.execute("INSERT INTO t VALUES (2,'y'),(1,'x')").unwrap();
    let rows = one(&mut db, "SELECT a, b FROM t ORDER BY a");
    assert_eq!(rows.len(), 2);
    assert!(matches!(rows[0][0], Value::Integer(1)));
    assert!(matches!(&rows[0][1], Value::Text(s) if s == "x"));
    assert!(matches!(rows[1][0], Value::Integer(2)));
}

#[test]
fn where_and_aggregate() {
    let mut db = Connection::open_in_memory().unwrap();
    db.execute("CREATE TABLE n(x INTEGER)").unwrap();
    db.execute("INSERT INTO n VALUES (1),(2),(3),(4)").unwrap();
    let rows = one(&mut db, "SELECT count(*), sum(x) FROM n WHERE x >= 2");
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], Value::Integer(3)));
    assert!(matches!(rows[0][1], Value::Integer(9)));
}

#[test]
fn null_semantics() {
    let mut db = Connection::open_in_memory().unwrap();
    db.execute("CREATE TABLE t(a INTEGER)").unwrap();
    db.execute("INSERT INTO t VALUES (1),(NULL)").unwrap();
    // NULL is excluded by `= 1` and by `<> 1`; count(*) sees both rows.
    assert!(matches!(one(&mut db, "SELECT count(*) FROM t")[0][0], Value::Integer(2)));
    assert_eq!(one(&mut db, "SELECT a FROM t WHERE a = 1").len(), 1);
    assert_eq!(one(&mut db, "SELECT a FROM t WHERE a <> 1").len(), 0);
}
