#![allow(clippy::unwrap_used)]
use scone_core::embed::{EmbeddingProvider, HashEmbedder};
use scone_core::index::vectors::VectorIndex;

#[test]
fn nearest_neighbor_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let e = HashEmbedder::new(64);
    let vecs = e
        .embed(&[
            "alpha beta gamma",
            "unrelated words entirely",
            "alpha beta delta",
        ])
        .unwrap();
    let mut idx = VectorIndex::open(dir.path(), 64).unwrap();
    idx.add(&[(1, &vecs[0]), (2, &vecs[1]), (3, &vecs[2])])
        .unwrap();
    idx.flush().unwrap();
    let q = e.embed(&["alpha beta gamma"]).unwrap();
    let hits = idx.search(&q[0], 2).unwrap();
    assert_eq!(hits[0].0, 1);
    assert!(hits[0].1 > hits[1].1, "top similarity should be highest");
    drop(idx);
    let idx = VectorIndex::open(dir.path(), 64).unwrap();
    let hits = idx.search(&q[0], 1).unwrap();
    assert_eq!(hits[0].0, 1);
}

#[test]
fn dim_mismatch_on_reopen_is_typed_and_names_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let e = HashEmbedder::new(32);
    let v = e.embed(&["something"]).unwrap();
    let mut idx = VectorIndex::open(dir.path(), 32).unwrap();
    idx.add(&[(7, &v[0])]).unwrap();
    idx.flush().unwrap();
    drop(idx);
    let err = VectorIndex::open(dir.path(), 64).unwrap_err();
    match err {
        scone_core::SconeError::Index(msg) => assert!(msg.contains("rebuild"), "{msg}"),
        other => panic!("expected Index error, got {other:?}"),
    }
}

#[test]
fn empty_index_searches_empty_and_wipe_clears() {
    let dir = tempfile::tempdir().unwrap();
    let e = HashEmbedder::new(16);
    let v = e.embed(&["x y z"]).unwrap();
    let mut idx = VectorIndex::open(dir.path(), 16).unwrap();
    assert!(idx.search(&v[0], 3).unwrap().is_empty());
    idx.add(&[(1, &v[0])]).unwrap();
    idx.wipe().unwrap();
    assert!(idx.search(&v[0], 3).unwrap().is_empty());
}
