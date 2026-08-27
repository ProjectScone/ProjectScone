#![allow(clippy::unwrap_used)]
use scone_core::Engine;

#[test]
fn open_creates_schema_and_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let e = Engine::open(dir.path()).unwrap();
    drop(e);
    let e = Engine::open(dir.path()).unwrap();
    assert_eq!(e.schema_version().unwrap(), 1);
    assert!(dir.path().join("scone.db").exists());
}
