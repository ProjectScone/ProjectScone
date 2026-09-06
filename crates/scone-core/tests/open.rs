#![allow(clippy::unwrap_used)]
use scone_core::Engine;
use scone_core::embed::HashEmbedder;

#[test]
fn open_creates_schema_and_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    drop(e);
    let e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert_eq!(e.schema_version().unwrap(), 5);
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
    assert_eq!(engine.schema_version().unwrap(), 5);
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

/// The v5 migration rebuilds the facts table to admit `proposed` and
/// `declined`. A v4 store is made by hand (the v4 shape, with a fact,
/// its provenance and a closed sibling), then opened: every row, id and
/// reference must survive, and the widened status must be accepted.
#[test]
fn widening_fact_statuses_keeps_every_row_and_reference() {
    use scone_core::auth;

    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "default", true).unwrap();
    let (episode_id, _) = engine
        .import_episode(
            &space,
            "note",
            "mark lives in lisbon",
            None,
            Some("2024-03-02T12:00:00.000Z"),
        )
        .unwrap();
    drop(engine);

    // Downgrade to the v4 shape: the old CHECK, the same rows, version 4.
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    raw.pragma_update(None, "foreign_keys", "OFF").unwrap();
    raw.execute_batch(
        "CREATE TABLE facts_v4 (
            id INTEGER PRIMARY KEY, space_id INTEGER NOT NULL REFERENCES spaces(id),
            subject_entity INTEGER NOT NULL REFERENCES entities(id),
            predicate TEXT NOT NULL, object TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            valid_from TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            valid_until TEXT,
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','closed','expired')),
            status_reason TEXT, last_accessed TEXT, access_count INTEGER NOT NULL DEFAULT 0);
         DROP TABLE facts;
         ALTER TABLE facts_v4 RENAME TO facts;
         INSERT INTO entities (id, canonical) VALUES (1, 'mark');
         INSERT INTO facts (id, space_id, subject_entity, predicate, object, confidence, valid_from, valid_until, status, status_reason)
             VALUES (7, 1, 1, 'lives_in', 'austin', 0.9, '2022-01-01T00:00:00.000Z', '2024-03-02T12:00:00.000Z', 'closed', 'superseded by fact 9'),
                    (9, 1, 1, 'lives_in', 'lisbon', 0.8, '2024-03-02T12:00:00.000Z', NULL, 'active', NULL);
         UPDATE meta SET value = '4' WHERE key = 'schema_version';",
    )
    .unwrap();
    raw.execute(
        "INSERT INTO fact_provenance (fact_id, episode_id) VALUES (9, ?1)",
        [episode_id],
    )
    .unwrap();
    drop(raw);

    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert_eq!(engine.schema_version().unwrap(), 5);
    let space = auth::resolve(&mut engine, "default", true).unwrap();
    let facts = engine.facts_list(&space, true).unwrap();
    assert_eq!(
        facts
            .iter()
            .map(|f| (f.fact_id, f.object.as_str(), f.status.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (7, "austin", "closed (superseded by fact 9)"),
            (9, "lisbon", "active")
        ],
        "rows, ids, statuses and reasons survive the rebuild"
    );
    assert_eq!(
        facts[0].valid_until.as_deref(),
        Some("2024-03-02T12:00:00.000Z")
    );
    assert_eq!(
        engine.facts_why(&space, 9).unwrap().len(),
        1,
        "provenance still points at the rebuilt table"
    );
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    raw.execute(
        "INSERT INTO facts (space_id, subject_entity, predicate, object, status) VALUES (1, 1, 'drinks', 'tea', 'proposed')",
        [],
    )
    .expect("the widened CHECK admits proposed");
    let n: i64 = raw
        .query_row("SELECT count(*) FROM sqlite_master WHERE type='index' AND name='facts_subject_predicate'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1, "the index is recreated on the new table");
}
