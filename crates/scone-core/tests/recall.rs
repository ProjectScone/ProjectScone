#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::{ContextPack, Engine, IngestInput, RecallOpts, auth};

fn setup(dir: &std::path::Path) -> (Engine, scone_core::auth::ScopedSpace) {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    for text in [
        "the rust borrow checker enforces ownership and lifetimes",
        "my sourdough starter doubles after six hours at room temp",
        "the meeting about quarterly planning moved to thursday",
        "tantivy gives lucene-class bm25 search to rust programs",
    ] {
        e.ingest(&space, IngestInput::Note { text: text.into() })
            .unwrap();
    }
    (e, space)
}

fn recall(e: &mut Engine, s: &scone_core::auth::ScopedSpace, q: &str) -> ContextPack {
    e.recall(s, q, &RecallOpts::default()).unwrap()
}

#[test]
fn finds_the_relevant_note_first() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = setup(dir.path());
    let pack = recall(&mut e, &space, "borrow checker ownership");
    assert!(!pack.items.is_empty());
    assert!(
        pack.items[0].text.contains("borrow checker"),
        "{}",
        pack.items[0].text
    );
}

#[test]
fn other_space_sees_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, _space) = setup(dir.path());
    let other = auth::resolve(&mut e, "other", true).unwrap();
    let pack = recall(&mut e, &other, "borrow checker");
    assert!(pack.items.is_empty());
}

#[test]
fn deleted_truth_is_never_served_even_if_indexed() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = setup(dir.path());
    let pack = recall(&mut e, &space, "sourdough starter");
    let episode_id = pack.items[0].episode_id;
    // Simulate truth deletion behind the indexes' back (bugs.md P-3).
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    raw.execute(
        "DELETE FROM distill_queue WHERE episode_id = ?1",
        [episode_id],
    )
    .unwrap();
    raw.execute("DELETE FROM chunks WHERE episode_id = ?1", [episode_id])
        .unwrap();
    raw.execute("DELETE FROM episodes WHERE id = ?1", [episode_id])
        .unwrap();
    drop(raw);
    let pack = recall(&mut e, &space, "sourdough starter");
    assert!(pack.items.iter().all(|i| i.episode_id != episode_id));
}

#[test]
fn budget_keeps_top_item_then_truncates() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = setup(dir.path());
    let opts = RecallOpts {
        limit: 10,
        budget_bytes: Some(1),
    };
    let pack = e.recall(&space, "rust search", &opts).unwrap();
    assert_eq!(pack.items.len(), 1);
}

#[test]
fn unparseable_query_degrades_instead_of_failing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = setup(dir.path());
    let pack = e
        .recall(&space, "AND ((( \"", &RecallOpts::default())
        .unwrap();
    assert!(!pack.degraded.is_empty());
}
