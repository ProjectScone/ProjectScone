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
    // Only active facts belong in a profile.
    e.facts_close(&space, profile.static_facts[0].fact_id, "moved away")
        .unwrap();
    let after = e.profile(&space, 5).unwrap();
    assert!(after.static_facts.iter().all(|f| f.object != "austin"));
}

#[test]
fn empty_space_yields_empty_profile() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "empty", true).unwrap();
    let profile = e.profile(&space, 5).unwrap();
    assert!(profile.static_facts.is_empty() && profile.dynamic.is_empty());
}
