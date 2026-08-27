#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

fn fact(s: &str, p: &str, o: &str) -> ExtractedFact {
    ExtractedFact {
        subject: s.into(),
        predicate: p.into(),
        object: o.into(),
        confidence: 0.9,
    }
}

fn setup(dir: &std::path::Path) -> (Engine, scone_core::auth::ScopedSpace, i64) {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "mark talked about package managers".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    (e, space, episode_id)
}

#[test]
fn facts_surface_for_entity_queries_and_reinforce() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "bun")])
        .unwrap();
    let pack = e
        .recall(&space, "what does mark prefer", &RecallOpts::default())
        .unwrap();
    assert_eq!(pack.facts.len(), 1);
    assert_eq!(pack.facts[0].object, "bun");
    e.recall(&space, "mark", &RecallOpts::default()).unwrap();
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    let (count, accessed): (i64, Option<String>) = db
        .query_row("SELECT access_count, last_accessed FROM facts", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(count, 2, "each recall reinforces");
    assert!(accessed.is_some());
}

#[test]
fn as_of_returns_what_was_true_then() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "pnpm")])
        .unwrap();
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "bun")])
        .unwrap();
    // Pin the intervals to controlled dates.
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    db.execute(
        "UPDATE facts SET valid_from='2026-01-01T00:00:00Z', valid_until='2026-06-01T00:00:00Z'
         WHERE object='pnpm'",
        [],
    )
    .unwrap();
    db.execute(
        "UPDATE facts SET valid_from='2026-06-01T00:00:00Z' WHERE object='bun'",
        [],
    )
    .unwrap();
    drop(db);
    let now = e
        .recall(&space, "mark prefers", &RecallOpts::default())
        .unwrap();
    assert_eq!(now.facts.len(), 1);
    assert_eq!(now.facts[0].object, "bun", "today, bun is the truth");
    let then = e
        .recall(
            &space,
            "mark prefers",
            &RecallOpts {
                as_of: Some("2026-03-15T00:00:00Z".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(then.facts.len(), 1);
    assert_eq!(then.facts[0].object, "pnpm", "in march, pnpm was the truth");
    assert_eq!(
        then.facts[0].status, "closed",
        "history is served from closed intervals"
    );
}

#[test]
fn facts_are_space_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "bun")])
        .unwrap();
    let other = auth::resolve(&mut e, "other", true).unwrap();
    let pack = e.recall(&other, "mark", &RecallOpts::default()).unwrap();
    assert!(pack.facts.is_empty());
}
