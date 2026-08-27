#![allow(clippy::unwrap_used)]
use proptest::prelude::*;
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, IngestInput, IngestOutcome, auth};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    /// Invariants I2, I3, I4 hold under arbitrary fact sequences.
    #[test]
    fn semantic_invariants_hold(seq in prop::collection::vec((0..2usize, 0..2usize, 0..3usize), 1..24)) {
        let dir = tempfile::tempdir().unwrap();
        let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(16))).unwrap();
        let space = auth::resolve(&mut e, "default", true).unwrap();
        let IngestOutcome::Ingested { episode_id, .. } = e
            .ingest(&space, IngestInput::Note { text: "prop seed".into() })
            .unwrap() else { panic!() };
        for (s, p, o) in &seq {
            e.apply_facts(&space, episode_id, &[ExtractedFact {
                subject: format!("subject-{s}"),
                predicate: format!("pred-{p}"),
                object: format!("object-{o}"),
                confidence: 0.5,
            }]).unwrap();
        }
        let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
        // I2: at most one active fact per (subject, predicate).
        let max_active: i64 = db.query_row(
            "SELECT coalesce(max(n), 0) FROM (
                SELECT count(*) AS n FROM facts
                WHERE status = 'active'
                GROUP BY space_id, subject_entity, predicate)",
            [], |r| r.get(0)).unwrap();
        prop_assert!(max_active <= 1, "I2 violated: {max_active} active for one key");
        // I3: closed facts keep both interval ends and a reason.
        let broken_closed: i64 = db.query_row(
            "SELECT count(*) FROM facts WHERE status = 'closed'
               AND (valid_until IS NULL OR status_reason IS NULL)",
            [], |r| r.get(0)).unwrap();
        prop_assert_eq!(broken_closed, 0, "I3 violated");
        // I4: every fact has provenance.
        let orphans: i64 = db.query_row(
            "SELECT count(*) FROM facts f
             WHERE NOT EXISTS (SELECT 1 FROM fact_provenance fp WHERE fp.fact_id = f.id)",
            [], |r| r.get(0)).unwrap();
        prop_assert_eq!(orphans, 0, "I4 violated");
    }
}
