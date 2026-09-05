#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A question about last week should find last week's memory.
//!
//! Explicit dates already influenced ranking. Relative references did
//! not, so "which book did I finish a week ago" was decided purely on
//! topical similarity, and the best-matching book from any week won.
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, RecallOpts, auth};

#[test]
fn a_relative_reference_favours_the_right_week() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();

    // Two equally good answers, one in the week asked about and one
    // not. The wrong week is stored first so that ties break in its
    // favour, which means the date bonus is the only thing that can
    // put the right week on top. Disable it and this test fails.
    e.import_episode(
        &space,
        "note",
        "finished reading Project Hail Mary",
        None,
        Some("2024-03-22T12:00:00.000Z"),
    )
    .unwrap();
    e.import_episode(
        &space,
        "note",
        "finished reading The Nightingale",
        None,
        Some("2024-03-08T12:00:00.000Z"),
    )
    .unwrap();

    // Asked on the 15th, "a week ago" points at the 8th.
    let opts = RecallOpts {
        limit: 2,
        as_of: Some("2024-03-15T00:00:00.000Z".into()),
        ..Default::default()
    };
    let pack = e
        .recall(&space, "which book did I finish a week ago", &opts)
        .unwrap();
    let top = &pack.items.first().expect("something comes back").text;
    assert!(
        top.contains("Nightingale"),
        "the week asked about must lead, got: {top}"
    );
}

/// The window must not become a filter. A question that mentions a
/// relative time but whose answer sits outside it still has to come
/// back, because "a week ago" in speech is approximate and a hard cut
/// would drop the very memory being asked for.
#[test]
fn a_relative_reference_does_not_hide_everything_else() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    for (day, text) in [
        ("2024-01-05", "adopted a golden retriever named Biscuit"),
        ("2024-03-08", "renewed the office lease"),
    ] {
        e.import_episode(
            &space,
            "note",
            text,
            None,
            Some(&format!("{day}T12:00:00.000Z")),
        )
        .unwrap();
    }
    let opts = RecallOpts {
        limit: 5,
        as_of: Some("2024-03-15T00:00:00.000Z".into()),
        ..Default::default()
    };
    let pack = e
        .recall(
            &space,
            "what did I name the dog I adopted a week ago",
            &opts,
        )
        .unwrap();
    assert!(
        pack.items.iter().any(|i| i.text.contains("Biscuit")),
        "a memory outside the window must still be reachable: {:?}",
        pack.items.iter().map(|i| &i.text).collect::<Vec<_>>()
    );
}
