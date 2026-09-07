#![allow(clippy::unwrap_used)]
//! Native source inventory survives reopen and a deleted cursor record.
use scone_core::{Engine, auth, embed::HashEmbedder};

#[test]
fn inventory_reopens_and_keeps_a_deleted_boundary_usable() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "alpha", true).unwrap();
    for text in ["first source", "second source", "third source"] {
        engine
            .import_episode(&space, "file", text, Some("retained.txt"), None)
            .unwrap();
    }
    let first = engine.source_page(&space, None, 2, Some("file")).unwrap();
    assert_eq!(first["next_before"], 2);
    drop(engine);
    // This is disposable test state, simulating a source removed between pages.
    let connection = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    connection
        .execute_batch("PRAGMA foreign_keys=ON; BEGIN; DELETE FROM chunks WHERE episode_id=2; DELETE FROM distill_queue WHERE episode_id=2; DELETE FROM episodes WHERE id=2; COMMIT;")
        .unwrap();
    drop(connection);
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "alpha", true).unwrap();
    let page = engine
        .source_page(&space, Some(2), 2, Some("file"))
        .unwrap();
    assert_eq!(page["items"].as_array().unwrap().len(), 1);
    assert_eq!(page["items"][0]["preview"], "first source");
    assert_eq!(page["has_more"], false);
    assert!(page["next_before"].is_null());
    assert!(engine.source_page(&space, None, 0, None).is_err());
    assert!(engine.source_page(&space, Some(-1), 25, None).is_err());
}

#[test]
fn every_source_row_says_where_the_queue_left_it() {
    // cited: a claim rests on it; parked: the queue gave up; done: visited,
    // nothing to cite; pending: the queue will visit it. The same words the
    // Python host uses, so a shared client reads one vocabulary.
    use scone_core::llm::ExtractedFact;
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "alpha", true).unwrap();
    let mut ids = Vec::new();
    for text in [
        "mark works at acme",
        "unreadable scan",
        "nothing here",
        "fresh note",
    ] {
        ids.push(
            engine
                .import_episode(&space, "file", text, Some("retained.txt"), None)
                .unwrap()
                .0,
        );
    }
    engine
        .apply_facts(
            &space,
            ids[0],
            &[ExtractedFact {
                subject: "mark".into(),
                predicate: "works_at".into(),
                object: "acme".into(),
                confidence: 1.0,
            }],
        )
        .unwrap();
    drop(engine);
    let connection = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    connection
        .execute(
            "UPDATE distill_queue SET state='failed', attempts=3, last_error='model replied with no JSON' WHERE episode_id=?1",
            [ids[1]],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE distill_queue SET state='done' WHERE episode_id=?1",
            [ids[2]],
        )
        .unwrap();
    drop(connection);
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "alpha", true).unwrap();
    let page = engine.source_page(&space, None, 10, None).unwrap();
    let status = |id: i64| {
        page["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["episode_id"] == id)
            .unwrap()
            .clone()
    };
    assert_eq!(status(ids[0])["status"], "cited");
    assert_eq!(status(ids[1])["status"], "parked");
    assert_eq!(
        status(ids[1])["parked_reason"],
        "model replied with no JSON"
    );
    assert_eq!(status(ids[2])["status"], "done");
    assert_eq!(status(ids[3])["status"], "pending");
    assert!(status(ids[3]).get("parked_reason").is_none());
}
