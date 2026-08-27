#![allow(clippy::unwrap_used)]
use scone_bench::{Report, parse_dataset, run_item};
use scone_core::Engine;
use scone_core::embed::HashEmbedder;
use scone_core::llm::FakeLlm;

#[test]
fn parses_the_fixture() {
    let raw = include_str!("fixtures/mini.json");
    let items = parse_dataset(raw).unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].question_id, "mini-1");
    assert_eq!(items[0].sessions.len(), 2);
    assert!(items[0].sessions[0][0].contains("Gaggia"));
}

#[test]
fn runs_an_item_and_scores_substring_match() {
    let raw = include_str!("fixtures/mini.json");
    let items = parse_dataset(raw).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    // FakeLlm answer() echoes the question and context length; force a
    // correct answer by programming extraction and answering ourselves is
    // out of scope — instead verify the pipeline: ingest happened, recall
    // returned the right session, and scoring logic works on both sides.
    engine.set_llm(Some(Box::new(FakeLlm::new(vec![]))));
    let outcome = run_item(&mut engine, &items[0], 0).unwrap();
    assert!(
        outcome.retrieved.contains("Gaggia"),
        "recall must surface the right session"
    );
    assert!(!outcome.correct, "fake answer cannot match");
    assert!(outcome.stored_bytes > 0 && outcome.retrieved_bytes > 0);
    // Context reduction is meaningful at corpus scale, not on a two-session
    // fixture where everything legitimately fits; its math is unit-tested
    // in report_aggregates.
}

#[test]
fn report_aggregates() {
    let mut report = Report::default();
    report.add(true, 100, 10);
    report.add(false, 100, 30);
    assert_eq!(report.total, 2);
    assert_eq!(report.correct, 1);
    assert!((report.accuracy() - 0.5).abs() < f64::EPSILON);
    assert!((report.context_reduction() - 0.8).abs() < 1e-9);
}
