#![allow(clippy::unwrap_used, clippy::expect_used)]
//! The two operators that answer from several episodes at once.
//!
//! Ordering by category and measuring a span both pick their own
//! members, which is exactly how the earlier ordering operator invented
//! answers. These tests are mostly about the refusals: a confident
//! wrong answer is worse than none, because it arrives stated as fact.
use scone_core::embed::OnnxEmbedder;
use scone_core::{Engine, auth};

/// These operators pick their own members by similarity, so they need
/// an embedder with real semantics. The hash embedder produces
/// near-orthogonal vectors, which put every candidate below the anchor
/// floor and made the refusal tests pass for the wrong reason.
fn store(dir: &std::path::Path, rows: &[(&str, &str)]) -> (Engine, auth::ScopedSpace) {
    let cache = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".scone");
    let mut e = Engine::open(dir, Box::new(OnnxEmbedder::new(&cache).unwrap())).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    for (day, text) in rows {
        e.import_episode(
            &space,
            "note",
            text,
            None,
            Some(&format!("{day}T12:00:00.000Z")),
        )
        .unwrap();
    }
    (e, space)
}

#[test]
fn a_span_needs_two_mentions_to_bound_it() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(
        dir.path(),
        &[(
            "2024-02-01",
            "started reading The Nightingale by Kristin Hannah",
        )],
    );
    // One mention cannot bound anything.
    let answer = e
        .answer_temporally(
            &space,
            "How many days did it take me to finish 'The Nightingale' by Kristin Hannah?",
            Some("2024-03-01T00:00:00Z"),
        )
        .unwrap();
    assert!(answer.is_none(), "one mention is not a span: {answer:?}");
}

#[test]
fn a_span_measures_first_mention_to_last() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(
        dir.path(),
        &[
            (
                "2024-02-01",
                "started reading The Nightingale by Kristin Hannah today",
            ),
            (
                "2024-02-15",
                "finished reading The Nightingale by Kristin Hannah today",
            ),
        ],
    );
    let answer = e
        .answer_temporally(
            &space,
            "How many days did it take me to finish 'The Nightingale' by Kristin Hannah?",
            Some("2024-03-01T00:00:00Z"),
        )
        .unwrap()
        .expect("two dated mentions bound a span");
    assert_eq!(answer.value, "14 days", "{}", answer.derivation);
    assert!(
        answer.derivation.contains("2024-02-01"),
        "{}",
        answer.derivation
    );
}

/// A question that says how many members it expects is a gift: finding
/// a different number proves the wrong things were found, and ordering
/// them anyway would be the old failure in a new operator.
#[test]
fn ordering_a_category_refuses_when_the_count_disagrees() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(
        dir.path(),
        &[
            ("2024-01-10", "visited the Museum of Modern Art with Sarah"),
            ("2024-02-20", "visited the Natural History Museum today"),
        ],
    );
    // The question expects six; the store holds two.
    let answer = e
        .answer_temporally(
            &space,
            "What is the order of the six museums I visited from earliest to latest?",
            Some("2024-03-01T00:00:00Z"),
        )
        .unwrap();
    assert!(
        answer.is_none(),
        "six asked, two found, so it must decline: {answer:?}"
    );
}

#[test]
fn ordering_a_category_puts_the_members_in_time_order() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(
        dir.path(),
        &[
            ("2024-02-20", "visited the Natural History Museum today"),
            ("2024-01-10", "visited the Museum of Modern Art today"),
        ],
    );
    let answer = e
        .answer_temporally(
            &space,
            "What is the order of the two museums I visited from earliest to latest?",
            Some("2024-03-01T00:00:00Z"),
        )
        .unwrap()
        .expect("two asked, two found");
    let modern = answer.value.find("Modern Art");
    let natural = answer.value.find("Natural History");
    assert!(
        modern < natural,
        "January must precede February: {}",
        answer.value
    );
}

/// Grounding must pick the closest match, not the best-ranked result.
/// Ranking folds in recency; grounding asks which episode a phrase
/// describes, and the answer does not become truer for being recent.
/// This is the two-episode store where the defect first showed: the
/// MoMA phrase ranked the later Met episode first, both anchors landed
/// on it, and the interval was declined. Grounding by similarity puts
/// each phrase on its own episode and the interval computes.
#[test]
fn anchors_ground_to_the_closest_episode_not_the_most_recent() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(
        dir.path(),
        &[
            (
                "2026-08-20",
                "I visited the Museum of Modern Art with Sarah today",
            ),
            (
                "2026-08-27",
                "went to the Ancient Civilizations exhibit at the Met this afternoon",
            ),
        ],
    );
    let answer = e
        .answer_temporally(
            &space,
            "How many days passed between my visit to the Museum of Modern Art \
             and the Ancient Civilizations exhibit?",
            Some("2026-09-01T00:00:00Z"),
        )
        .unwrap()
        .expect("two distinct anchors must ground to two distinct episodes");
    assert_eq!(answer.value, "7 days", "{}", answer.derivation);
    assert!(
        answer.derivation.contains("episode 1"),
        "{}",
        answer.derivation
    );
    assert!(
        answer.derivation.contains("episode 2"),
        "{}",
        answer.derivation
    );
}
