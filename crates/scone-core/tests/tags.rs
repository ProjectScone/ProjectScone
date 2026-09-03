#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

#[test]
fn fresh_store_is_current_schema_with_tag_tables() {
    let dir = tempfile::tempdir().unwrap();
    let e = engine(dir.path());
    assert_eq!(e.schema_version().unwrap(), 4);
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    for t in ["tags", "episode_tags"] {
        let n: i64 = raw
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [t],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "missing {t}");
    }
}

#[test]
fn tags_scope_recall_to_focused_context() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let IngestOutcome::Ingested {
        episode_id: work, ..
    } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "quarterly report deadline is friday".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    let IngestOutcome::Ingested {
        episode_id: hobby, ..
    } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "the sourdough deadline is saturday bake".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    e.tag_episode(&space, work, &["work", "q3"]).unwrap();
    e.tag_episode(&space, hobby, &["baking"]).unwrap();

    let all = e
        .recall(&space, "deadline", &RecallOpts::default())
        .unwrap();
    assert_eq!(all.items.len(), 2, "untagged recall sees everything");

    let focused = e
        .recall(
            &space,
            "deadline",
            &RecallOpts {
                tags: vec!["work".into()],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(focused.items.len(), 1, "tag focus narrows");
    assert!(focused.items[0].text.contains("quarterly"));

    // AND semantics: both tags must be present.
    let strict = e
        .recall(
            &space,
            "deadline",
            &RecallOpts {
                tags: vec!["work".into(), "q3".into()],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(strict.items.len(), 1);
    let none = e
        .recall(
            &space,
            "deadline",
            &RecallOpts {
                tags: vec!["work".into(), "baking".into()],
                ..Default::default()
            },
        )
        .unwrap();
    assert!(none.items.is_empty(), "AND semantics: no episode has both");
}

#[test]
fn tags_list_with_counts_and_are_space_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "tagged note".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    e.tag_episode(&space, episode_id, &["work"]).unwrap();
    e.tag_episode(&space, episode_id, &["work"]).unwrap(); // idempotent
    let tags = e.tags_list(&space).unwrap();
    assert_eq!(tags, vec![("work".to_owned(), 1)]);
    let other = auth::resolve(&mut e, "other", true).unwrap();
    assert!(e.tags_list(&other).unwrap().is_empty());
    // Bounded input (P-4): garbage tag names are typed errors.
    assert!(e.tag_episode(&space, episode_id, &[""]).is_err());
    let long = "x".repeat(65);
    assert!(e.tag_episode(&space, episode_id, &[long.as_str()]).is_err());
}
