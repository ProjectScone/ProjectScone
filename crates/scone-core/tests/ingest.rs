#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, IngestOutcome, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

#[test]
fn ingest_note_stores_episode_chunks_and_bumps_revision() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let before = e.space_revision(&space).unwrap();
    let out = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "remember: the borrow checker is a friend".into(),
            },
        )
        .unwrap();
    let IngestOutcome::Ingested { episode_id, chunks } = out else {
        panic!("expected Ingested, got {out:?}");
    };
    assert!(chunks >= 1);
    assert_eq!(e.space_revision(&space).unwrap(), before + 1);
    let (content, kind) = e.episode_content(episode_id).unwrap();
    assert_eq!(content, "remember: the borrow checker is a friend");
    assert_eq!(kind, "note");
}

#[test]
fn duplicate_note_is_deduplicated_and_revision_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let first = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "same".into(),
            },
        )
        .unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = first else {
        panic!("expected Ingested");
    };
    let rev = e.space_revision(&space).unwrap();
    let second = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "same".into(),
            },
        )
        .unwrap();
    assert!(matches!(
        second,
        IngestOutcome::Deduplicated { episode_id: id } if id == episode_id
    ));
    assert_eq!(e.space_revision(&space).unwrap(), rev);
}

#[test]
fn same_content_in_other_space_is_not_a_duplicate() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let a = auth::resolve(&mut e, "a", true).unwrap();
    let b = auth::resolve(&mut e, "b", true).unwrap();
    e.ingest(
        &a,
        IngestInput::Note {
            text: "same".into(),
        },
    )
    .unwrap();
    let out = e
        .ingest(
            &b,
            IngestInput::Note {
                text: "same".into(),
            },
        )
        .unwrap();
    assert!(matches!(out, IngestOutcome::Ingested { .. }));
}

#[test]
fn file_ingest_reads_utf8_and_rejects_binary_typed() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let good = dir.path().join("note.md");
    std::fs::write(&good, "# heading\n\nbody text").unwrap();
    let out = e
        .ingest(&space, IngestInput::File { path: good.clone() })
        .unwrap();
    assert!(matches!(out, IngestOutcome::Ingested { .. }));
    let bad = dir.path().join("blob.bin");
    std::fs::write(&bad, [0xff, 0xfe, 0x00, 0x9f]).unwrap();
    let err = e
        .ingest(&space, IngestInput::File { path: bad })
        .unwrap_err();
    assert!(matches!(err, scone_core::SconeError::InvalidInput(_)));
}

#[test]
fn empty_note_is_rejected_typed() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let err = e
        .ingest(&space, IngestInput::Note { text: "".into() })
        .unwrap_err();
    assert!(matches!(err, scone_core::SconeError::InvalidInput(_)));
}

#[test]
fn chunk_target_is_tunable() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    e.set_chunk_target(100);
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let text = "para one is long enough to matter for chunking purposes here.\n\n\
para two is also long enough to matter for chunking purposes here.\n\n\
para three is likewise long enough to matter for chunking here.";
    let out = e
        .ingest(&space, IngestInput::Note { text: text.into() })
        .unwrap();
    let IngestOutcome::Ingested { chunks, .. } = out else {
        panic!()
    };
    assert!(
        chunks >= 2,
        "small target must split paragraphs, got {chunks}"
    );
}
