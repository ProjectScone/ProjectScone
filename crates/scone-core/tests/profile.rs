#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, IngestInput, IngestOutcome, auth};

#[test]
fn profile_has_stable_facts_and_recent_activity() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "first note about setup".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    e.ingest(
        &space,
        IngestInput::Note {
            text: "latest note about the demo".into(),
        },
    )
    .unwrap();
    e.apply_facts(
        &space,
        episode_id,
        &[
            ExtractedFact {
                subject: "mark".into(),
                predicate: "lives_in".into(),
                object: "austin".into(),
                confidence: 0.95,
            },
            ExtractedFact {
                subject: "mark".into(),
                predicate: "tried".into(),
                object: "matcha".into(),
                confidence: 0.3,
            },
        ],
    )
    .unwrap();
    let profile = e.profile(&space, 5).unwrap();
    assert!(!profile.static_facts.is_empty());
    assert_eq!(
        profile.static_facts[0].object, "austin",
        "confidence orders the identity"
    );
    assert!(!profile.dynamic.is_empty());
    assert!(
        profile.dynamic[0].contains("latest note"),
        "dynamic leads with the newest"
    );
    let recent: Vec<&str> = profile.recent.iter().map(|r| r.excerpt.as_str()).collect();
    let dynamic: Vec<&str> = profile.dynamic.iter().map(String::as_str).collect();
    assert_eq!(recent, dynamic, "recent is dynamic with its evidence");
    assert_eq!(
        profile.recent[1].episode_id, episode_id,
        "the older excerpt names the first episode"
    );
    assert!(profile.recent[0].episode_id > episode_id);
    assert!(
        profile.recent.iter().all(|r| !r.created_at.is_empty()),
        "each carries its episode's timestamp"
    );
    // Only active facts belong in a profile.
    e.facts_close(&space, profile.static_facts[0].fact_id, "moved away")
        .unwrap();
    let after = e.profile(&space, 5).unwrap();
    assert!(after.static_facts.iter().all(|f| f.object != "austin"));
}

/// One episode of `space` at `created_at`, taught its facts; returns the episode id.
fn taught(
    e: &mut Engine,
    space: &auth::ScopedSpace,
    text: &str,
    created_at: &str,
    facts: &[ExtractedFact],
) -> i64 {
    let (episode, _) = e
        .import_episode(space, "note", text, None, Some(created_at))
        .unwrap();
    e.apply_facts(space, episode, facts).unwrap();
    episode
}

fn fact(subject: &str, predicate: &str, object: &str) -> ExtractedFact {
    ExtractedFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        confidence: 0.9,
    }
}

/// Eligibility is time and space: a claim whose validity has not begun
/// and a claim of another space stay out of static_facts; another
/// space's episodes stay out of recent.
#[test]
fn a_profile_holds_only_claims_that_hold_now_in_its_space() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let other = auth::resolve(&mut e, "other", true).unwrap();
    let now_episode = taught(
        &mut e,
        &space,
        "mark lives in austin",
        "2024-01-01T00:00:00.000Z",
        &[fact("mark", "lives_in", "austin")],
    );
    let future_episode = taught(
        &mut e,
        &space,
        "mark will move to lisbon",
        "2999-01-01T00:00:00.000Z",
        &[fact("mark", "moves_to", "lisbon")],
    );
    taught(
        &mut e,
        &other,
        "someone else works at acme",
        "2024-01-02T00:00:00.000Z",
        &[fact("mark", "works_at", "acme")],
    );
    let profile = e.profile(&space, 5).unwrap();
    let objects: Vec<&str> = profile
        .static_facts
        .iter()
        .map(|f| f.object.as_str())
        .collect();
    assert_eq!(
        objects,
        vec!["austin"],
        "a claim not yet valid and another space's claim stay out"
    );
    let recent: Vec<i64> = profile.recent.iter().map(|r| r.episode_id).collect();
    assert_eq!(
        recent,
        vec![future_episode, now_episode],
        "episodes are not validity-gated, but another space's stay out"
    );
}

/// Recent is by the episode's own time, not by when it was added, on
/// both engines: a backfill sorts where it happened.
#[test]
fn a_backfilled_episode_sorts_where_it_happened() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let first = taught(
        &mut e,
        &space,
        "the newer note",
        "2024-02-01T00:00:00.000Z",
        &[],
    );
    let backfilled = taught(
        &mut e,
        &space,
        "the backfilled note",
        "2023-06-01T00:00:00.000Z",
        &[],
    );
    assert!(backfilled > first, "ids say when it was added");
    let profile = e.profile(&space, 5).unwrap();
    let recent: Vec<i64> = profile.recent.iter().map(|r| r.episode_id).collect();
    assert_eq!(
        recent,
        vec![first, backfilled],
        "order says when it happened"
    );
    assert_eq!(profile.dynamic[0], "the newer note");
}

#[test]
fn empty_space_yields_empty_profile() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "empty", true).unwrap();
    let profile = e.profile(&space, 5).unwrap();
    assert!(
        profile.static_facts.is_empty() && profile.dynamic.is_empty() && profile.recent.is_empty()
    );
}
