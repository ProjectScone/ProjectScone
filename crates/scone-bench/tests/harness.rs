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

#[test]
fn llm_judge_accepts_paraphrase_and_rejects_wrong() {
    use scone_bench::judge_correct;
    use scone_core::llm::FakeLlm;
    let yes = FakeLlm::new(vec![]).with_answer("YES — same city.");
    assert!(
        judge_correct(
            &yes,
            "where does the user live?",
            "austin",
            "They moved to Austin, Texas."
        )
        .unwrap()
    );
    let no = FakeLlm::new(vec![]).with_answer("NO");
    assert!(!judge_correct(&no, "where does the user live?", "austin", "Denver.").unwrap());
}

#[test]
fn typed_judges_select_by_question_type_and_parse_json() {
    use scone_bench::{judge_correct_typed, judge_prompt_for};
    use scone_core::llm::FakeLlm;
    assert!(judge_prompt_for("temporal-reasoning").contains("off-by-one"));
    assert!(judge_prompt_for("abstention-style").contains("abstain"));
    assert!(judge_prompt_for("knowledge-update").contains("updated answer"));
    assert!(judge_prompt_for("single-session-preference").contains("rubric"));
    let yes = FakeLlm::new(vec![])
        .with_answer(r#"{"score": 1, "label": "correct", "explanation": "same"}"#);
    assert!(
        judge_correct_typed(&yes, "multi-session", "q", "austin", "they moved to Austin").unwrap()
    );
    let no = FakeLlm::new(vec![])
        .with_answer(r#"{"score": 0, "label": "incorrect", "explanation": "different"}"#);
    assert!(!judge_correct_typed(&no, "multi-session", "q", "austin", "Denver").unwrap());
    // Non-JSON output falls back to yes/no prefix, conservatively.
    let messy = FakeLlm::new(vec![]).with_answer("yes, that matches");
    assert!(judge_correct_typed(&messy, "multi-session", "q", "a", "a").unwrap());
}

#[test]
fn question_date_reaches_the_answer_call() {
    use scone_core::embed::HashEmbedder;
    use scone_core::llm::FakeLlm;
    let raw = r#"[{"question_id":"d1","question_type":"temporal-reasoning",
        "question":"how many days between the trips?","answer":"3",
        "question_date":"2023/05/20 (Sat) 02:21",
        "haystack_sessions":[[{"role":"user","content":"first trip monday, second trip thursday"}]],
        "haystack_dates":["2023/05/01 (Mon) 10:00"],
        "haystack_session_ids":["s1"],"answer_session_ids":["s1"]}]"#;
    let items = scone_bench::parse_dataset(raw).unwrap();
    assert_eq!(items[0].question_date, "2023-05-20T02:21:00Z");
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let fake = FakeLlm::new(vec![]);
    engine.set_llm(Some(Box::new(FakeLlm::new(vec![]))));
    let _ = fake; // the engine's own fake echoes; verify via outcome text
    let outcome = scone_bench::run_item(&mut engine, &items[0], 0).unwrap();
    assert!(
        outcome.model_answer.contains("2023-05-20"),
        "the answer call must see the question date: {}",
        outcome.model_answer
    );
}
