#![allow(clippy::unwrap_used)]
//! The snapshot shows where a claim came from, or says it could not.
//!
//! Codex found the Python half of this while building the depth view. The
//! Rust half samples the newest episodes, so a claim whose source is older
//! than that sample lost its `source_of` edge silently, and the node bound
//! dropped whatever sorted late by key rather than whatever mattered least.
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, auth};

fn fact(subject: &str, predicate: &str, object: &str) -> ExtractedFact {
    ExtractedFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        confidence: 0.9,
    }
}

fn edges_of<'a>(graph: &'a serde_json::Value, kind: &str) -> Vec<(&'a str, &'a str)> {
    graph["edges"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == kind)
        .map(|e| {
            (
                e["source"].as_str().unwrap_or(""),
                e["target"].as_str().unwrap_or(""),
            )
        })
        .collect()
}

fn has_node(graph: &serde_json::Value, id: &str) -> bool {
    graph["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n["id"] == id)
}

#[test]
fn a_claim_shows_the_old_episode_it_came_from() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let (old, _) = e
        .import_episode(&space, "note", "ana moved to lisbon in march", None, None)
        .unwrap();
    e.apply_facts(&space, old, &[fact("ana", "moved_to", "lisbon")])
        .unwrap();
    let claim = e.facts_list(&space, true).unwrap()[0].fact_id;
    // Newer episodes crowd the sample the snapshot takes.
    for i in 0..8 {
        e.import_episode(&space, "note", &format!("something newer {i}"), None, None)
            .unwrap();
    }

    let graph = e.evidence_graph(&space, 6).unwrap();
    assert!(
        has_node(&graph, &format!("episode:{old}")),
        "the source is drawn even though it is not among the newest: {graph}"
    );
    assert!(
        edges_of(&graph, "source_of").contains(&(
            format!("episode:{old}").as_str(),
            format!("claim:{claim}").as_str()
        )),
        "and the claim still points at it"
    );
}

#[test]
fn what_the_bound_left_out_is_counted_rather_than_quietly_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    for i in 0..12 {
        let (episode, _) = e
            .import_episode(&space, "note", &format!("source number {i}"), None, None)
            .unwrap();
        e.apply_facts(
            &space,
            episode,
            &[fact(
                &format!("subject{i}"),
                "came_from",
                &format!("note{i}"),
            )],
        )
        .unwrap();
    }

    let small = e.evidence_graph(&space, 4).unwrap();
    assert_eq!(small["truncated"], true);
    assert!(
        small["provenance_omitted"].as_u64().unwrap() > 0,
        "a snapshot that could not hold both ends of an edge says so: {small}"
    );
    // Structure survives the bound; text is what goes first.
    let kinds: Vec<&str> = small["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["kind"].as_str().unwrap_or(""))
        .collect();
    assert!(
        kinds.contains(&"episode") && kinds.contains(&"claim"),
        "episodes and claims are kept before passages: {kinds:?}"
    );
}

#[test]
fn a_source_that_is_gone_is_counted_apart_from_one_that_did_not_fit() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let (gone, _) = e
        .import_episode(&space, "note", "a note that will be removed", None, None)
        .unwrap();
    e.apply_facts(
        &space,
        gone,
        &[fact("one", "came_from", "the removed note")],
    )
    .unwrap();
    // The episode row goes while the claim and its provenance stay. The
    // foreign key normally prevents exactly this, which is the point: the
    // count is what reports the state if a store ever reaches it, by a
    // restore, a repair or a hand-edited file, rather than the graph
    // quietly drawing a claim with nothing behind it.
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    raw.execute_batch("PRAGMA foreign_keys = OFF").unwrap();
    raw.execute("DELETE FROM episodes WHERE id = ?1", [gone])
        .unwrap();
    drop(raw);

    let graph = e.evidence_graph(&space, 50).unwrap();
    assert_eq!(
        graph["provenance_missing"], 1,
        "the source is named as gone, not as out of view: {graph}"
    );
    assert_eq!(
        graph["provenance_omitted"], 0,
        "and nothing was cut for room"
    );
    assert!(
        has_node(&graph, "claim:1"),
        "the claim stands; its source is what went"
    );
}
