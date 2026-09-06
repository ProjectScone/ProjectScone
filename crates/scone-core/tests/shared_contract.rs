#![allow(clippy::unwrap_used)]
//! Literal cross-product expectations, not equality between two engines.
//! Hash embeddings support structural tests here, never semantic-quality claims.
use scone_core::{Engine, auth, chunker, embed::HashEmbedder};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/episodes-v1.json")).unwrap()
}

#[test]
fn shared_episodes_preserve_evidence_and_scope_local_dedup() {
    // Removing source/date export, normalizing content, or global dedup breaks this.
    let corpus = fixture();
    let records = corpus["episodes"].as_array().unwrap();
    let data = records
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    for name in ["alpha", "beta"] {
        let space = auth::resolve(&mut engine, name, true).unwrap();
        assert_eq!(engine.import_jsonl(&space, &data).unwrap().episodes, 3);
        let repeat = engine.import_jsonl(&space, &data).unwrap();
        assert_eq!((repeat.episodes, repeat.deduplicated), (0, 3));
        let exported: Vec<Value> = engine
            .export_jsonl(&space)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(exported.len(), 3);
        for expected in records {
            let actual = exported
                .iter()
                .find(|r| r["content"] == expected["content"])
                .unwrap();
            for field in ["type", "kind", "content", "source", "created_at"] {
                assert_eq!(actual[field], expected[field], "{name}: {field}");
            }
        }
    }
}

#[test]
fn shared_unicode_span_uses_utf8_bytes() {
    // Code-point end=3 would split the emoji; the hand-counted byte end is 7.
    let corpus = fixture();
    let case = &corpus["span"];
    let content = case["content"].as_str().unwrap();
    let spans = chunker::chunk_text(content, 700);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].start, case["start"].as_u64().unwrap() as usize);
    assert_eq!(spans[0].end, case["end"].as_u64().unwrap() as usize);
    assert_eq!(&content[spans[0].start..spans[0].end], content);
}
