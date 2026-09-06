#![allow(clippy::unwrap_used)]
//! Inferred-fact review: an extracted fact below the engine's confidence
//! gate is a proposal. It is stored with its provenance and kept out of
//! every reading surface until a person approves it; approval places it
//! in the ledger exactly as an above-gate fact would have been placed.
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, RecallOpts, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

fn fact(subject: &str, predicate: &str, object: &str, confidence: f32) -> ExtractedFact {
    ExtractedFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        confidence,
    }
}

fn episode(e: &mut Engine, space: &auth::ScopedSpace, text: &str, day: &str) -> i64 {
    e.import_episode(
        space,
        "note",
        text,
        None,
        Some(&format!("{day}T12:00:00.000Z")),
    )
    .unwrap()
    .0
}

fn statuses(e: &Engine, space: &auth::ScopedSpace) -> Vec<(i64, String)> {
    e.facts_list(space, true)
        .unwrap()
        .into_iter()
        .map(|f| (f.fact_id, f.status))
        .collect()
}

#[test]
fn below_the_gate_a_fact_is_proposed_and_invisible_to_readers() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    e.set_propose_below(Some(0.7)).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let ep = episode(
        &mut e,
        &space,
        "mark moved to lisbon in march",
        "2024-03-02",
    );
    let report = e
        .apply_facts(
            &space,
            ep,
            &[
                fact("mark", "lives_in", "lisbon", 0.55),
                fact("mark", "drinks", "tea", 0.9),
            ],
        )
        .unwrap();
    assert_eq!((report.added, report.proposed, report.closed), (1, 1, 0));

    let pending = e.facts_pending(&space).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(
        (pending[0].object.as_str(), pending[0].status.as_str()),
        ("lisbon", "proposed")
    );
    assert_eq!(
        e.facts_why(&space, pending[0].fact_id).unwrap().len(),
        1,
        "a proposal keeps its provenance"
    );

    // Not on any reading surface.
    let recalled = e
        .recall(&space, "where does mark live", &RecallOpts::default())
        .unwrap();
    assert!(
        recalled.facts.iter().all(|f| f.object != "lisbon"),
        "{:?}",
        recalled.facts
    );
    assert!(
        e.facts_about(&space, "mark")
            .unwrap()
            .iter()
            .all(|f| f.object != "lisbon")
    );
    assert!(
        e.profile(&space, 5)
            .unwrap()
            .static_facts
            .iter()
            .all(|f| f.object != "lisbon")
    );
    assert_eq!(e.facts_list(&space, false).unwrap().len(), 1, "active only");
    assert_eq!(
        e.facts_list(&space, true).unwrap().len(),
        2,
        "all shows the proposal"
    );
}

#[test]
fn approval_places_the_fact_in_the_ledger_as_if_it_had_cleared_the_gate() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let austin = episode(&mut e, &space, "mark lives in austin", "2022-01-01");
    e.apply_facts(&space, austin, &[fact("mark", "lives_in", "austin", 0.9)])
        .unwrap();
    e.set_propose_below(Some(0.7)).unwrap();
    let lisbon = episode(&mut e, &space, "mark moved to lisbon", "2024-03-02");
    e.apply_facts(&space, lisbon, &[fact("mark", "lives_in", "lisbon", 0.6)])
        .unwrap();
    assert_eq!(
        statuses(&e, &space),
        vec![(1, "active".into()), (2, "proposed".into())]
    );

    let closed = e.facts_approve(&space, 2).unwrap();
    assert_eq!(closed, 1, "austin is superseded on approval, not before");
    let listed = e.facts_list(&space, true).unwrap();
    assert_eq!(listed[0].status, "closed (superseded by fact 2)");
    assert_eq!(
        listed[0].valid_until.as_deref(),
        Some("2024-03-02T12:00:00.000Z")
    );
    assert_eq!(listed[1].status, "active (approved)");
    let recalled = e
        .recall(&space, "where does mark live", &RecallOpts::default())
        .unwrap();
    assert_eq!(
        recalled
            .facts
            .iter()
            .map(|f| f.object.as_str())
            .collect::<Vec<_>>(),
        vec!["lisbon"]
    );

    // Approving a proposal that is older than the current answer closes
    // the proposal itself: history the moment it is accepted.
    let berlin = episode(&mut e, &space, "mark once lived in berlin", "2019-06-01");
    e.apply_facts(&space, berlin, &[fact("mark", "lives_in", "berlin", 0.5)])
        .unwrap();
    assert_eq!(e.facts_approve(&space, 3).unwrap(), 0);
    let listed = e.facts_list(&space, true).unwrap();
    assert_eq!(listed[2].status, "closed (superseded by fact 2)");
    assert_eq!(listed[1].status, "active (approved)", "lisbon still holds");
    assert!(e.facts_approve(&space, 3).is_err(), "not proposed any more");
}

#[test]
fn a_declined_fact_stays_declined_when_restated() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    e.set_propose_below(Some(0.7)).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let ep = episode(
        &mut e,
        &space,
        "mark's favourite colour is green",
        "2024-01-01",
    );
    e.apply_facts(
        &space,
        ep,
        &[fact("mark", "favourite_colour", "green", 0.4)],
    )
    .unwrap();
    e.facts_decline(&space, 1, "a guess from a joke").unwrap();
    assert_eq!(
        e.facts_list(&space, true).unwrap()[0].status,
        "declined (a guess from a joke)"
    );
    assert!(e.facts_pending(&space).unwrap().is_empty());

    let again = episode(&mut e, &space, "green again, they said", "2024-02-01");
    let report = e
        .apply_facts(
            &space,
            again,
            &[fact("mark", "favourite_colour", "green", 0.65)],
        )
        .unwrap();
    assert_eq!(
        (report.added, report.proposed, report.deduplicated),
        (0, 0, 1)
    );
    let facts = e.facts_list(&space, true).unwrap();
    assert_eq!(facts.len(), 1, "no second row");
    assert_eq!(facts[0].status, "declined (a guess from a joke)");
    assert!(
        (facts[0].confidence - 0.65).abs() < 1e-6,
        "strengthened, still declined"
    );
    assert!(
        e.facts_decline(&space, 1, "twice").is_err(),
        "only a proposal can be declined"
    );

    // A restated proposal stays a proposal too; repetition is evidence
    // for the reviewer, not approval.
    let ep2 = episode(&mut e, &space, "mark lives in lisbon", "2024-03-02");
    e.apply_facts(&space, ep2, &[fact("mark", "lives_in", "lisbon", 0.5)])
        .unwrap();
    let ep3 = episode(&mut e, &space, "lisbon, as mark keeps saying", "2024-04-02");
    let report = e
        .apply_facts(&space, ep3, &[fact("mark", "lives_in", "lisbon", 0.95)])
        .unwrap();
    assert_eq!(report.deduplicated, 1);
    let pending = e.facts_pending(&space).unwrap();
    assert_eq!(pending.len(), 1);
    assert!((pending[0].confidence - 0.95).abs() < 1e-6);
}

#[test]
fn without_a_gate_every_fact_is_active_and_the_gate_is_validated() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    assert_eq!(e.propose_below(), None);
    let ep = episode(&mut e, &space, "mark lives in lisbon", "2024-03-02");
    let report = e
        .apply_facts(&space, ep, &[fact("mark", "lives_in", "lisbon", 0.1)])
        .unwrap();
    assert_eq!((report.added, report.proposed), (1, 0));
    assert!(e.set_propose_below(Some(1.5)).is_err());
    assert!(e.set_propose_below(Some(-0.1)).is_err());
    e.set_propose_below(Some(1.0)).unwrap();
    assert_eq!(e.propose_below(), Some(1.0));
}
