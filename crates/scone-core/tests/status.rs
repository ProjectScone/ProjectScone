#![allow(clippy::unwrap_used)]
//! Backlog counts belong to a space. `status()` is the operator's view of
//! the whole store, which is what a local CLI should print; a key that
//! names one space must never learn how much work another space has.
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, auth};

#[test]
fn a_key_counts_its_own_backlog_while_the_operator_sees_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let alice = auth::resolve(&mut e, "alice", true).unwrap();
    let bob = auth::resolve(&mut e, "bob", true).unwrap();
    for (space, text) in [(&alice, "alice one"), (&bob, "bob one"), (&bob, "bob two")] {
        e.ingest(space, IngestInput::Note { text: text.into() })
            .unwrap();
    }
    // One of bob's is beyond saving, so the failed column means something.
    let raw = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    raw.execute(
        "UPDATE distill_queue SET state = 'failed' WHERE episode_id IN
           (SELECT e.id FROM episodes e JOIN spaces s ON s.id = e.space_id
            WHERE s.name = 'bob' ORDER BY e.id LIMIT 1)",
        [],
    )
    .unwrap();
    drop(raw);

    let mine = e.distill_backlog(&alice).unwrap();
    assert_eq!(
        (mine.pending, mine.failed),
        (1, 0),
        "a scoped key sees its own work only"
    );
    let theirs = e.distill_backlog(&bob).unwrap();
    assert_eq!((theirs.pending, theirs.failed), (1, 1));
    let report = e.status().unwrap();
    assert_eq!(
        (report.pending_distill, report.failed_distill),
        (2, 1),
        "the operator's view is the whole store, and says so"
    );
}
