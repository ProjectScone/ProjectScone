#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, RecallOpts, auth};

fn ingest_three(e: &mut Engine) -> scone_core::auth::ScopedSpace {
    let space = auth::resolve(e, "default", true).unwrap();
    for text in [
        "the rust borrow checker enforces ownership",
        "sourdough starter needs daily feeding",
        "tantivy provides bm25 full text search",
    ] {
        e.ingest(&space, IngestInput::Note { text: text.into() })
            .unwrap();
    }
    space
}

#[test]
fn rebuild_recovers_destroyed_indexes_from_truth() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    ingest_three(&mut e);
    drop(e);
    // Destroy the derived indexes behind the engine's back.
    std::fs::remove_dir_all(dir.path().join("fts")).unwrap();
    std::fs::remove_file(dir.path().join("vectors").join("vectors.usearch")).unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let pack = e
        .recall(&space, "sourdough", &RecallOpts::default())
        .unwrap();
    assert!(pack.items.is_empty(), "destroyed indexes should miss");
    let report = e.doctor_rebuild().unwrap();
    assert_eq!(report.episodes, 3);
    assert_eq!(report.chunks, 3);
    assert_eq!(
        report.reembedded, 0,
        "blobs match dim; no re-embedding needed"
    );
    let pack = e
        .recall(&space, "sourdough", &RecallOpts::default())
        .unwrap();
    assert!(pack.items[0].text.contains("sourdough"));
}

#[test]
fn embedder_swap_is_refused_then_repaired() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    ingest_three(&mut e);
    drop(e);
    let err = Engine::open(dir.path(), Box::new(HashEmbedder::new(128))).unwrap_err();
    assert!(err.to_string().contains("rebuild"), "{err}");
    let mut e = Engine::open_for_repair(dir.path(), Box::new(HashEmbedder::new(128))).unwrap();
    let report = e.doctor_rebuild().unwrap();
    assert_eq!(report.reembedded, 3, "dim changed; all chunks re-embed");
    drop(e);
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(128))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let pack = e
        .recall(&space, "borrow checker", &RecallOpts::default())
        .unwrap();
    assert!(pack.items[0].text.contains("borrow"));
}

#[test]
fn rebuild_preserves_multibyte_chunk_text() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let text = format!("{} das Gedächtnis — la mémoire — 記憶", "é".repeat(300));
    e.ingest(&space, IngestInput::Note { text }).unwrap();
    e.doctor_rebuild().unwrap();
    let pack = e.recall(&space, "mémoire", &RecallOpts::default()).unwrap();
    assert!(pack.items.iter().any(|i| i.text.contains("mémoire")));
}
