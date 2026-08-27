#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, auth};

#[test]
fn resolve_creates_once_and_validates() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let s1 = auth::resolve(&mut e, "default", true).unwrap();
    let s2 = auth::resolve(&mut e, "default", true).unwrap();
    assert_eq!(s1.name(), s2.name());
    assert!(auth::resolve(&mut e, "missing", false).is_err());
    assert!(auth::resolve(&mut e, "BAD NAME!", true).is_err());
    assert!(auth::resolve(&mut e, "", true).is_err());
    let long = "x".repeat(65);
    assert!(auth::resolve(&mut e, &long, true).is_err());
}
