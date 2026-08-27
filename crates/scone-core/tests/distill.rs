#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::llm::{ExtractedFact, FakeLlm};
use scone_core::{Engine, IngestInput, auth};

fn fact(s: &str, p: &str, o: &str) -> ExtractedFact {
    ExtractedFact {
        subject: s.into(),
        predicate: p.into(),
        object: o.into(),
        confidence: 0.8,
    }
}

fn engine(dir: &std::path::Path) -> (Engine, scone_core::auth::ScopedSpace) {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    (e, space)
}

#[test]
fn distill_drains_pending_and_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = engine(dir.path());
    e.ingest(
        &space,
        IngestInput::Note {
            text: "mark prefers bun".into(),
        },
    )
    .unwrap();
    e.ingest(
        &space,
        IngestInput::Note {
            text: "mark lives in austin".into(),
        },
    )
    .unwrap();
    e.set_llm(Some(Box::new(FakeLlm::new(vec![fact(
        "mark", "prefers", "bun",
    )]))));
    let r = e.distill(&space, 10).unwrap();
    assert_eq!(r.processed, 2);
    assert_eq!(r.facts_added, 1, "same fact from both episodes dedupes");
    assert_eq!(r.failed, 0);
    let again = e.distill(&space, 10).unwrap();
    assert_eq!(again.processed, 0, "queue drained; distill is idempotent");
}

#[test]
fn distill_without_llm_is_a_typed_loud_error() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = engine(dir.path());
    e.ingest(
        &space,
        IngestInput::Note {
            text: "note".into(),
        },
    )
    .unwrap();
    let err = e.distill(&space, 10).unwrap_err();
    assert!(err.to_string().contains("no LLM configured"), "{err}");
}

#[test]
fn provider_failure_is_recorded_and_retried_after_fix() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = engine(dir.path());
    e.ingest(
        &space,
        IngestInput::Note {
            text: "flaky".into(),
        },
    )
    .unwrap();
    e.set_llm(Some(Box::new(FakeLlm::failing("model overloaded"))));
    let r = e.distill(&space, 10).unwrap();
    assert_eq!((r.processed, r.failed), (0, 1));
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    let (state, attempts, err): (String, i64, Option<String>) = db
        .query_row(
            "SELECT state, attempts, last_error FROM distill_queue",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(state, "pending", "one failure keeps it pending for retry");
    assert_eq!(attempts, 1);
    assert!(
        err.unwrap().contains("model overloaded"),
        "failure evidence kept (P-2)"
    );
    // Fix the provider; the episode distills on the next run.
    e.set_llm(Some(Box::new(FakeLlm::new(vec![fact("a", "b", "c")]))));
    let r = e.distill(&space, 10).unwrap();
    assert_eq!((r.processed, r.facts_added), (1, 1));
}

#[test]
fn three_failures_park_the_episode_as_failed() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = engine(dir.path());
    e.ingest(
        &space,
        IngestInput::Note {
            text: "cursed".into(),
        },
    )
    .unwrap();
    e.set_llm(Some(Box::new(FakeLlm::failing("boom"))));
    for _ in 0..3 {
        e.distill(&space, 10).unwrap();
    }
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    let state: String = db
        .query_row("SELECT state FROM distill_queue", [], |r| r.get(0))
        .unwrap();
    assert_eq!(state, "failed", "parked, never deleted (P-2)");
    assert_eq!(
        e.distill(&space, 10).unwrap().processed,
        0,
        "failed rows are not retried implicitly"
    );
}

#[test]
fn malformed_extracted_facts_are_skipped_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = engine(dir.path());
    e.ingest(
        &space,
        IngestInput::Note {
            text: "note with messy extraction".into(),
        },
    )
    .unwrap();
    e.set_llm(Some(Box::new(FakeLlm::new(vec![
        ExtractedFact {
            subject: "mark".into(),
            predicate: "".into(),
            object: "junk".into(),
            confidence: 0.5,
        },
        ExtractedFact {
            subject: "".into(),
            predicate: "likes".into(),
            object: "".into(),
            confidence: 0.5,
        },
        ExtractedFact {
            subject: "mark".into(),
            predicate: "uses".into(),
            object: "scone".into(),
            confidence: 0.9,
        },
    ]))));
    let r = e.distill(&space, 10).unwrap();
    assert_eq!(r.processed, 1, "the episode still processes");
    assert_eq!(r.facts_added, 1, "the valid fact lands");
    assert_eq!(r.failed, 0, "malformed facts are skipped, not batch-fatal");
    let facts = e.facts_list(&space, false).unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].object, "scone");
}
