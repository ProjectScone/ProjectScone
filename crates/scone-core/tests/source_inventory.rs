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
