#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

fn fact(s: &str, p: &str, o: &str, conf: f32) -> ExtractedFact {
    ExtractedFact {
        subject: s.into(),
        predicate: p.into(),
        object: o.into(),
        confidence: conf,
    }
}

fn setup(dir: &std::path::Path) -> (Engine, scone_core::auth::ScopedSpace, i64) {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "seed for decay tests".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    (e, space, episode_id)
}

fn age_fact(dir: &std::path::Path, object: &str, days: i64) {
    let db = rusqlite::Connection::open(dir.join("scone.db")).unwrap();
    db.execute(
        "UPDATE facts SET valid_from = strftime('%Y-%m-%dT%H:%M:%fZ','now', ?1)
         WHERE object = ?2",
        rusqlite::params![format!("-{days} days"), object],
    )
    .unwrap();
}

#[test]
fn stale_unaccessed_low_confidence_facts_expire_with_reason() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(
        &space,
        ep,
        &[
            fact("mark", "tried", "matcha", 0.4),
            fact("mark", "lives_in", "austin", 0.95),
        ],
    )
    .unwrap();
    age_fact(dir.path(), "matcha", 120);
    age_fact(dir.path(), "austin", 120);
    let expired = e.decay_facts(&space, 90).unwrap();
    assert_eq!(expired, 1, "only the low-confidence unaccessed fact decays");
    let all = e.facts_list(&space, true).unwrap();
    let matcha = all.iter().find(|f| f.object == "matcha").unwrap();
    assert!(matcha.status.starts_with("expired"), "{}", matcha.status);
    assert!(
        matcha.status.contains("unaccessed"),
        "decay carries its reason: {}",
        matcha.status
    );
    assert!(
        matcha.valid_until.is_some(),
        "expiry closes the interval (I3)"
    );
    let austin = all.iter().find(|f| f.object == "austin").unwrap();
    assert_eq!(austin.status, "active", "high confidence survives");
}

#[test]
fn recalled_facts_are_reinforced_against_decay() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(&space, ep, &[fact("mark", "tried", "matcha", 0.4)])
        .unwrap();
    age_fact(dir.path(), "matcha", 120);
    // A recall touches the fact: access_count bumps, last_accessed = now.
    e.recall(&space, "matcha", &RecallOpts::default()).unwrap();
    let expired = e.decay_facts(&space, 90).unwrap();
    assert_eq!(expired, 0, "recently recalled facts never decay");
}

#[test]
fn fresh_facts_do_not_decay() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(&space, ep, &[fact("mark", "tried", "matcha", 0.1)])
        .unwrap();
    assert_eq!(e.decay_facts(&space, 90).unwrap(), 0);
}
