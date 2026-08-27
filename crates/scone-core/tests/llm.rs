#![allow(clippy::unwrap_used)]
use scone_core::Engine;
use scone_core::embed::HashEmbedder;
use scone_core::llm::{ExtractedFact, FakeLlm, LlmProvider};

#[test]
fn fake_llm_returns_programmed_facts_and_logs_calls() {
    let fake = FakeLlm::new(vec![ExtractedFact {
        subject: "mark".into(),
        predicate: "prefers".into(),
        object: "bun".into(),
        confidence: 0.9,
    }]);
    let facts = fake.extract_facts("mark said he prefers bun").unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].predicate, "prefers");
    assert_eq!(fake.calls(), vec!["mark said he prefers bun".to_owned()]);
    assert!(fake.answer("q", "ctx").unwrap().contains("q"));
}

#[test]
fn failing_fake_returns_typed_error() {
    let fake = FakeLlm::failing("model overloaded");
    let err = fake.extract_facts("text").unwrap_err();
    assert!(err.to_string().contains("model overloaded"));
}

#[test]
fn engine_llm_is_none_until_set() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    assert!(!e.has_llm());
    e.set_llm(Some(Box::new(FakeLlm::new(vec![]))));
    assert!(e.has_llm());
    e.set_llm(None);
    assert!(!e.has_llm());
}
