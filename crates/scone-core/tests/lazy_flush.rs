#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, RecallOpts, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

#[test]
fn recall_sees_all_ingests_immediately() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    for i in 0..3 {
        e.ingest(
            &space,
            IngestInput::Note {
                text: format!("flush test note {i} alpha"),
            },
        )
        .unwrap();
    }
    let pack = e
        .recall(
            &space,
            "flush test alpha",
            &RecallOpts {
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        pack.items.len(),
        3,
        "lazy flush must not hide fresh ingests"
    );
}

#[test]
fn reopen_after_drop_sees_everything() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = engine(dir.path());
        let space = auth::resolve(&mut e, "default", true).unwrap();
        e.ingest(
            &space,
            IngestInput::Note {
                text: "persisted across drop".into(),
            },
        )
        .unwrap();
        // no recall — flush must happen on drop
    }
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let pack = e
        .recall(&space, "persisted across drop", &RecallOpts::default())
        .unwrap();
    assert_eq!(pack.items.len(), 1);
}

#[test]
fn catch_up_on_open_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = engine(dir.path());
        let space = auth::resolve(&mut e, "default", true).unwrap();
        for i in 0..3 {
            e.ingest(
                &space,
                IngestInput::Note {
                    text: format!("catchup note {i} beta"),
                },
            )
            .unwrap();
        }
        e.recall(&space, "catchup beta", &RecallOpts::default())
            .unwrap(); // flushed
    }
    // Simulate a crash where the flush landed but the high-water meta write
    // was lost: catch-up must re-index the tail without duplicating.
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    db.execute("UPDATE meta SET value = '1' WHERE key = 'indexed_max'", [])
        .unwrap();
    drop(db);
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let pack = e
        .recall(
            &space,
            "catchup beta",
            &RecallOpts {
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(pack.items.len(), 3, "no duplicates, no gaps after catch-up");
}
