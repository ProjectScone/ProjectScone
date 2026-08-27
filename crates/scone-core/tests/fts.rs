#![allow(clippy::unwrap_used)]
use scone_core::index::fts::FtsIndex;

#[test]
fn fts_scopes_by_space_and_persists() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = FtsIndex::open(dir.path()).unwrap();
    idx.add(&[
        (1, 10, "the borrow checker enforces ownership"),
        (2, 10, "sourdough starter needs feeding"),
        (3, 20, "the borrow checker in space twenty"),
    ])
    .unwrap();
    let hits = idx.search(10, "borrow checker", 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 1);
    let hits = idx.search(20, "borrow", 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 3);
    let hits = idx.search(10, "sourdough", 5).unwrap();
    assert_eq!(hits[0].0, 2);
    drop(idx);
    let idx = FtsIndex::open(dir.path()).unwrap();
    let hits = idx.search(10, "ownership", 5).unwrap();
    assert_eq!(hits[0].0, 1);
}

#[test]
fn malformed_query_is_typed_error_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let idx = FtsIndex::open(dir.path()).unwrap();
    let err = idx.search(1, "AND ((( \"", 5).unwrap_err();
    assert!(matches!(err, scone_core::SconeError::InvalidInput(_)));
}

#[test]
fn wipe_empties_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = FtsIndex::open(dir.path()).unwrap();
    idx.add(&[(1, 1, "hello world")]).unwrap();
    idx.wipe().unwrap();
    assert!(idx.search(1, "hello", 5).unwrap().is_empty());
}
