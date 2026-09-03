#![allow(clippy::unwrap_used)]
use scone_core::Engine;
use scone_core::embed::HashEmbedder;

#[test]
fn open_creates_schema_and_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    drop(e);
    let e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert_eq!(e.schema_version().unwrap(), 4);
    assert!(dir.path().join("scone.db").exists());
}

/// The v4 migration rebuilds the episodes table to widen its kind
/// constraint. A rebuild that loses rows, or leaves the tables that
/// point at episodes dangling, would be far worse than the constraint
/// it fixes.
#[test]
fn widening_episode_kinds_keeps_every_row_and_reference() {
    use scone_core::IngestInput;
    use scone_core::auth;

    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "default", true).unwrap();
    engine
        .ingest(
            &space,
            IngestInput::Note {
                text: "a note from before the migration".into(),
            },
        )
        .unwrap();
    let (episode_id, _) = engine
        .import_episode(
            &space,
            "connector",
            "pulled from a service",
            Some("u"),
            None,
        )
        .unwrap();
    engine.tag_episode(&space, episode_id, &["github"]).unwrap();
    let counts = |e: &mut Engine| -> (i64, i64) {
        let s = e.status().unwrap();
        s.spaces
            .iter()
            .find(|s| s.name == "default")
            .map(|s| (s.episodes, s.chunks))
            .unwrap_or((0, 0))
    };
    let before = counts(&mut engine);
    drop(engine);

    // Reopening runs migrations again; they must be idempotent.
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert_eq!(engine.schema_version().unwrap(), 4);
    assert_eq!(
        counts(&mut engine),
        before,
        "the rebuild must not drop episodes or chunks"
    );
    let space = auth::resolve(&mut engine, "default", true).unwrap();
    let tags = engine.tags_list(&space).unwrap();
    assert!(
        tags.iter()
            .any(|(name, count)| name == "github" && *count == 1),
        "tags must still point at the rebuilt episodes: {tags:?}"
    );
}
