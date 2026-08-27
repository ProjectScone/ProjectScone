#![allow(clippy::unwrap_used)]
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, IngestInput, IngestOutcome, auth};

fn fact(s: &str, p: &str, o: &str) -> ExtractedFact {
    ExtractedFact {
        subject: s.into(),
        predicate: p.into(),
        object: o.into(),
        confidence: 0.8,
    }
}

fn setup(dir: &std::path::Path) -> (Engine, scone_core::auth::ScopedSpace, i64) {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "seed episode".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    (e, space, episode_id)
}

fn raw(dir: &std::path::Path) -> rusqlite::Connection {
    rusqlite::Connection::open(dir.join("scone.db")).unwrap()
}

#[test]
fn apply_creates_fact_with_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    let r = e
        .apply_facts(&space, ep, &[fact("Mark", "prefers", "bun")])
        .unwrap();
    assert_eq!((r.added, r.closed), (1, 0));
    let db = raw(dir.path());
    let (canonical, prov): (String, i64) = db
        .query_row(
            "SELECT en.canonical,
                    (SELECT count(*) FROM fact_provenance fp WHERE fp.fact_id = f.id)
             FROM facts f JOIN entities en ON en.id = f.subject_entity",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(canonical, "mark", "subjects canonicalize to lowercase");
    assert_eq!(prov, 1, "invariant I4: provenance required");
}

#[test]
fn exact_duplicate_collapses_and_accumulates_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep1) = setup(dir.path());
    let IngestOutcome::Ingested {
        episode_id: ep2, ..
    } = e
        .ingest(
            &space,
            IngestInput::Note {
                text: "second episode".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    e.apply_facts(&space, ep1, &[fact("mark", "prefers", "bun")])
        .unwrap();
    let r = e
        .apply_facts(&space, ep2, &[fact("Mark", "prefers", "bun")])
        .unwrap();
    assert_eq!((r.added, r.closed, r.deduplicated), (0, 0, 1));
    let db = raw(dir.path());
    let facts: i64 = db
        .query_row("SELECT count(*) FROM facts", [], |r| r.get(0))
        .unwrap();
    let prov: i64 = db
        .query_row("SELECT count(*) FROM fact_provenance", [], |r| r.get(0))
        .unwrap();
    assert_eq!(facts, 1);
    assert_eq!(prov, 2, "duplicate adds provenance, not a new fact");
}

#[test]
fn contradiction_closes_old_interval_with_reason() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "pnpm")])
        .unwrap();
    let r = e
        .apply_facts(&space, ep, &[fact("mark", "prefers", "bun")])
        .unwrap();
    assert_eq!((r.added, r.closed), (1, 1));
    let db = raw(dir.path());
    let (status, until, reason): (String, Option<String>, Option<String>) = db
        .query_row(
            "SELECT status, valid_until, status_reason FROM facts WHERE object = 'pnpm'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(status, "closed");
    assert!(
        until.is_some(),
        "invariant I3: closed facts keep both interval ends"
    );
    assert!(
        reason.unwrap().contains("superseded"),
        "closure carries a reason"
    );
    let active: i64 = db
        .query_row(
            "SELECT count(*) FROM facts WHERE status = 'active'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active, 1, "invariant I2");
}

#[test]
fn alias_merges_subjects() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.add_entity_alias("sconeuser00", "mark").unwrap();
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "bun")])
        .unwrap();
    let r = e
        .apply_facts(&space, ep, &[fact("sconeuser00", "prefers", "zig")])
        .unwrap();
    assert_eq!(
        (r.added, r.closed),
        (1, 1),
        "alias resolves to same entity, so this contradicts"
    );
}

#[test]
fn facts_about_resolves_aliases_and_scopes() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space, ep) = setup(dir.path());
    e.add_entity_alias("sconeuser00", "mark").unwrap();
    e.apply_facts(&space, ep, &[fact("mark", "prefers", "bun")])
        .unwrap();
    e.apply_facts(&space, ep, &[fact("sourdough", "needs", "feeding")])
        .unwrap();
    let about = e.facts_about(&space, "SconeUser00").unwrap();
    assert_eq!(about.len(), 1, "alias + case fold resolve to mark");
    assert_eq!(about[0].object, "bun");
    let other = auth::resolve(&mut e, "other", true).unwrap();
    assert!(
        e.facts_about(&other, "mark").unwrap().is_empty(),
        "space-scoped"
    );
    let none = e.facts_about(&space, "nobody").unwrap();
    assert!(none.is_empty());
}
