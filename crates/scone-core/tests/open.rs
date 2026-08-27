#![allow(clippy::unwrap_used)]
use scone_core::Engine;
use scone_core::embed::HashEmbedder;

#[test]
fn open_creates_schema_and_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    drop(e);
    let e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert_eq!(e.schema_version().unwrap(), 2);
    assert!(dir.path().join("scone.db").exists());
}
