#![allow(clippy::unwrap_used)]
//! Deleting a space: the preview says what would go, the deed removes it
//! and marks the space deleted, and the name cannot be re-created.
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, RecallOpts, SconeError, auth};

fn fact(subject: &str, predicate: &str, object: &str) -> ExtractedFact {
    ExtractedFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        confidence: 0.9,
    }
}

#[test]
fn deleting_a_space_removes_everything_and_keeps_it_gone() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "team", true).unwrap();
    let other = auth::resolve(&mut e, "other", true).unwrap();
    let (first, _) = e
        .import_episode(
            &space,
            "note",
            "acme is headquartered in lisbon",
            None,
            Some("2024-01-01T00:00:00.000Z"),
        )
        .unwrap();
    e.import_episode(
        &space,
        "note",
        "acme moved its office in march",
        None,
        Some("2024-02-01T00:00:00.000Z"),
    )
    .unwrap();
    e.apply_facts(
        &space,
        first,
        &[
            fact("acme", "based_in", "lisbon"),
            fact("mark", "works_at", "acme"),
        ],
    )
    .unwrap();
    let facts = e.facts_list(&space, true).unwrap();
    e.link_facts(
        &space,
        facts[1].fact_id,
        facts[0].fact_id,
        "supports",
        None,
        None,
    )
    .unwrap();
    let (neighbour, _) = e
        .import_episode(
            &other,
            "note",
            "the neighbour keeps its own note",
            None,
            None,
        )
        .unwrap();
    e.apply_facts(&other, neighbour, &[fact("neighbour", "keeps", "note")])
        .unwrap();

    let preview = e.space_impact(&space).unwrap();
    assert_eq!(
        (
            preview.episodes,
            preview.facts,
            preview.links,
            preview.tombstones
        ),
        (2, 2, 1, 0)
    );
    assert!(preview.chunks >= 2 && preview.deleted_at.is_none());
    assert_eq!(
        e.facts_list(&space, true).unwrap().len(),
        2,
        "a preview removes nothing"
    );

    let receipt = e.delete_space(&space).unwrap();
    assert_eq!(
        (
            receipt.episodes,
            receipt.chunks,
            receipt.facts,
            receipt.links
        ),
        (
            preview.episodes,
            preview.chunks,
            preview.facts,
            preview.links
        )
    );
    assert!(receipt.deleted_at.is_some());
    assert!(
        matches!(
            auth::resolve(&mut e, "team", true),
            Err(SconeError::NotFound(_))
        ),
        "a deleted space is not re-created by its name"
    );
    assert!(e.facts_list(&space, true).unwrap().is_empty());
    assert!(
        e.recall(&space, "acme lisbon", &RecallOpts::default())
            .unwrap()
            .items
            .is_empty()
    );
    assert!(e.profile(&space, 5).unwrap().recent.is_empty());
    assert!(matches!(
        e.delete_space(&space),
        Err(SconeError::NotFound(_))
    ));
    // The neighbour is untouched.
    assert_eq!(e.facts_list(&other, true).unwrap().len(), 1);
    assert!(!e.profile(&other, 5).unwrap().recent.is_empty());
    assert!(
        !e.recall(&other, "neighbour note", &RecallOpts::default())
            .unwrap()
            .items
            .is_empty()
    );
}
