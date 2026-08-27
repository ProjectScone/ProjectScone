#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

#[test]
fn fresh_store_is_schema_v2_with_semantic_tables() {
    let dir = tempfile::tempdir().unwrap();
    let e = engine(dir.path());
    assert_eq!(e.schema_version().unwrap(), 2);
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    for table in [
        "facts",
        "entities",
        "entity_aliases",
        "fact_provenance",
        "distill_queue",
    ] {
        let n: i64 = raw
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [table],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "missing table {table}");
    }
}

#[test]
fn ingest_enqueues_episode_for_distillation() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    e.ingest(
        &space,
        IngestInput::Note {
            text: "mark prefers bun over pnpm".into(),
        },
    )
    .unwrap();
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    let (state, attempts): (String, i64) = raw
        .query_row("SELECT state, attempts FROM distill_queue", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(state, "pending");
    assert_eq!(attempts, 0);
}

#[test]
fn v1_store_migrates_and_backfills_queue() {
    let dir = tempfile::tempdir().unwrap();
    // Fabricate a v1 store: current schema minus v2, version pinned to 1.
    {
        let mut e = engine(dir.path());
        let space = auth::resolve(&mut e, "default", true).unwrap();
        e.ingest(
            &space,
            IngestInput::Note {
                text: "old note".into(),
            },
        )
        .unwrap();
    }
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    raw.execute_batch(
        "DROP TABLE distill_queue; DROP TABLE fact_provenance; DROP TABLE facts;
         DROP TABLE entity_aliases; DROP TABLE entities;
         UPDATE meta SET value='1' WHERE key='schema_version';",
    )
    .unwrap();
    drop(raw);
    let e = engine(dir.path());
    assert_eq!(e.schema_version().unwrap(), 2);
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    let pending: i64 = raw
        .query_row(
            "SELECT count(*) FROM distill_queue WHERE state='pending'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pending, 1, "existing episodes must be back-enqueued");
}
