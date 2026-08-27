#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, RecallOpts, auth};

#[test]
fn directory_scan_ingests_recursively_and_rescans_cheaply() {
    let data = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(notes.path().join("sub/.hidden")).unwrap();
    std::fs::create_dir_all(notes.path().join("node_modules")).unwrap();
    std::fs::write(notes.path().join("a.md"), "alpha note about rust").unwrap();
    std::fs::write(notes.path().join("sub/b.txt"), "beta note about baking").unwrap();
    std::fs::write(
        notes.path().join("sub/.hidden/c.md"),
        "hidden should be skipped",
    )
    .unwrap();
    std::fs::write(notes.path().join("node_modules/d.md"), "dep dir skipped").unwrap();
    std::fs::write(notes.path().join("blob.bin"), [0xffu8, 0xfe, 0x00, 0x9f]).unwrap();
    std::fs::write(notes.path().join("huge.txt"), "x".repeat(2_000_000)).unwrap();

    let mut e = Engine::open(data.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let report = e.ingest_directory(&space, notes.path(), 1_000_000).unwrap();
    assert_eq!(report.ingested, 2, "only a.md and sub/b.txt qualify");
    assert_eq!(
        report.skipped, 2,
        "binary and oversized are counted, not silent"
    );

    let pack = e
        .recall(&space, "baking beta", &RecallOpts::default())
        .unwrap();
    assert!(pack.items[0].source.as_deref().unwrap().ends_with("b.txt"));

    let again = e.ingest_directory(&space, notes.path(), 1_000_000).unwrap();
    assert_eq!(again.ingested, 0);
    assert_eq!(
        again.deduplicated, 2,
        "unchanged files dedup by content hash"
    );

    std::fs::write(notes.path().join("a.md"), "alpha note about rust, edited").unwrap();
    let after_edit = e.ingest_directory(&space, notes.path(), 1_000_000).unwrap();
    assert_eq!(
        after_edit.ingested, 1,
        "edits are new episodes (append-only truth)"
    );
}
