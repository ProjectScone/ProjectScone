#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, RecallOpts, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

#[test]
fn second_process_degrades_to_read_only_search() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = engine(dir.path());
    let w_space = auth::resolve(&mut writer, "default", true).unwrap();
    writer
        .ingest(
            &w_space,
            IngestInput::Note {
                text: "shared truth is readable".into(),
            },
        )
        .unwrap();
    // Flush so the second opener can see committed index state.
    writer
        .recall(&w_space, "warmup", &RecallOpts::default())
        .unwrap();

    // Second engine on the same data dir while the first holds the write
    // lock: reads must work, writes must fail loudly, nothing corrupts.
    let mut reader = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert!(reader.is_read_only(), "writer lock is held; degrade loudly");
    let r_space = auth::resolve(&mut reader, "default", true).unwrap();
    let pack = reader
        .recall(&r_space, "shared truth", &RecallOpts::default())
        .unwrap();
    assert!(!pack.items.is_empty(), "read-only search still works");
    let err = reader
        .ingest(
            &r_space,
            IngestInput::Note {
                text: "should be refused".into(),
            },
        )
        .unwrap_err();
    assert!(err.to_string().contains("read-only"), "{err}");

    // The writer keeps working throughout.
    writer
        .ingest(
            &w_space,
            IngestInput::Note {
                text: "writer unaffected".into(),
            },
        )
        .unwrap();
}

#[test]
fn lock_releases_when_the_writer_closes() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut w = engine(dir.path());
        let space = auth::resolve(&mut w, "default", true).unwrap();
        w.ingest(
            &space,
            IngestInput::Note {
                text: "first owner".into(),
            },
        )
        .unwrap();
    }
    let mut next = engine(dir.path());
    assert!(!next.is_read_only(), "lock released on drop");
    let space = auth::resolve(&mut next, "default", true).unwrap();
    next.ingest(
        &space,
        IngestInput::Note {
            text: "second owner writes fine".into(),
        },
    )
    .unwrap();
}
