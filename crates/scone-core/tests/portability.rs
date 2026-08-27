#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

#[test]
fn export_import_round_trips_episodes_and_fact_history() {
    let a_dir = tempfile::tempdir().unwrap();
    let mut a = engine(a_dir.path());
    let space = auth::resolve(&mut a, "default", true).unwrap();
    a.ingest(
        &space,
        IngestInput::Note {
            text: "mark prefers pnpm for now".into(),
        },
    )
    .unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = a
        .ingest(
            &space,
            IngestInput::Note {
                text: "correction: mark prefers bun".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    a.add_entity_alias("sconeuser00", "mark").unwrap();
    a.apply_facts(
        &space,
        episode_id,
        &[ExtractedFact {
            subject: "mark".into(),
            predicate: "prefers".into(),
            object: "pnpm".into(),
            confidence: 0.7,
        }],
    )
    .unwrap();
    a.apply_facts(
        &space,
        episode_id,
        &[ExtractedFact {
            subject: "mark".into(),
            predicate: "prefers".into(),
            object: "bun".into(),
            confidence: 0.9,
        }],
    )
    .unwrap();
    let exported = a.export_jsonl(&space).unwrap();
    assert!(
        exported.lines().count() >= 5,
        "episodes + alias + facts:\n{exported}"
    );

    let b_dir = tempfile::tempdir().unwrap();
    let mut b = engine(b_dir.path());
    let b_space = auth::resolve(&mut b, "default", true).unwrap();
    let report = b.import_jsonl(&b_space, &exported).unwrap();
    assert_eq!(report.episodes, 2);
    assert_eq!(report.facts, 2);
    assert_eq!(report.aliases, 1);

    // Episodic memory searchable in the new store.
    let pack = b
        .recall(&b_space, "prefers bun", &RecallOpts::default())
        .unwrap();
    assert!(!pack.items.is_empty());
    // Fact history preserved: closed pnpm fact with reason, active bun fact.
    let all = b.facts_list(&b_space, true).unwrap();
    assert_eq!(all.len(), 2);
    assert!(
        all.iter()
            .any(|f| f.object == "pnpm" && f.status.starts_with("closed"))
    );
    assert!(
        all.iter()
            .any(|f| f.object == "bun" && f.status == "active")
    );
    // Provenance survived via episode hashes (I4 in the new store too).
    let bun = all.iter().find(|f| f.object == "bun").unwrap();
    assert!(!b.facts_why(&b_space, bun.fact_id).unwrap().is_empty());

    // Idempotent: importing again changes nothing.
    let again = b.import_jsonl(&b_space, &exported).unwrap();
    assert_eq!(again.episodes, 0, "dedup by content hash");
    assert_eq!(again.facts, 0, "facts dedup by identity tuple");
    assert_eq!(b.facts_list(&b_space, true).unwrap().len(), 2);
}

#[test]
fn import_rejects_garbage_lines_typed() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let err = e.import_jsonl(&space, "not json at all\n").unwrap_err();
    assert!(matches!(err, scone_core::SconeError::InvalidInput(_)));
}
