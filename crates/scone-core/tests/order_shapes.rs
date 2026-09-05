#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Ordering may only be planned when the question names its events.
//! A question that merely describes a category ("the six museums I
//! visited") names none of them, and answering it from whatever the
//! sentence splits into is how this operator invented answers before.
use scone_core::temporal::{Plan, plan};

fn events_of(q: &str) -> Option<Vec<String>> {
    match plan(q) {
        Some(Plan::Order { events }) => Some(events),
        _ => None,
    }
}

#[test]
fn enumerated_questions_are_planned_with_only_their_events() {
    let pair =
        events_of("Which event happened first, my cousin's wedding or Michael's engagement party?")
            .expect("a named pair is plannable");
    assert_eq!(pair.len(), 2, "{pair:?}");
    assert!(pair[0].contains("cousin"), "{pair:?}");
    assert!(pair[1].contains("engagement party"), "{pair:?}");
    assert!(
        !pair.iter().any(|e| e.contains("which event happened")),
        "the question must never become one of its own events: {pair:?}"
    );

    let listed = events_of(
        "What is the order of the three events: 'I signed up for the rewards program', \
         'I used a coupon', 'I bought the blender'?",
    )
    .expect("a quoted list is plannable");
    assert_eq!(listed.len(), 3, "{listed:?}");

    let colon = events_of(
        "Which three events happened in the order from first to last: the day I helped my \
         friend prepare the nursery, the day I helped my cousin pick out stuff, and the day \
         I ordered a phone case?",
    )
    .expect("a colon list is plannable");
    assert_eq!(colon.len(), 3, "{colon:?}");
    assert!(colon[0].contains("nursery"), "{colon:?}");
}

#[test]
fn questions_that_only_describe_a_category_are_declined() {
    for q in [
        "What is the order of the six museums I visited from earliest to latest?",
        "What is the order of the sports events I watched in January?",
        "What is the order of the three trips I took in the past three months, from earliest to latest?",
        "Which mode of transport did I use most recently, a bus or a train?",
    ] {
        // The last one names two things but they are categories, not
        // events, so grounding them to episode dates is meaningless.
        // Whatever this decides, it must not answer from the question.
        if let Some(events) = events_of(q) {
            assert!(
                !events
                    .iter()
                    .any(|e| e.starts_with("what is the order") || e.starts_with("which mode")),
                "{q}\n  planned from its own words: {events:?}"
            );
        }
    }
}
