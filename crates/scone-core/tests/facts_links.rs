#![allow(clippy::unwrap_used)]
//! Typed relations between facts, the same contract the Python engine
//! holds: same space only, one link per (from, to, kind), no self link,
//! no dependency cycle among extends/derived_from, a quote checked
//! against the episode it names, and the link kept when nothing else is.
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, SconeError, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

fn fact(subject: &str, predicate: &str, object: &str) -> ExtractedFact {
    ExtractedFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        confidence: 1.0,
    }
}

/// One held fact in `space`, taught by its own episode; returns (fact, episode).
fn held(e: &mut Engine, space: &auth::ScopedSpace, text: &str, f: ExtractedFact) -> (i64, i64) {
    let ep = e
        .import_episode(space, "note", text, None, Some("2024-03-02T12:00:00.000Z"))
        .unwrap()
        .0;
    e.apply_facts(space, ep, &[f]).unwrap();
    let id = e
        .facts_list(space, true)
        .unwrap()
        .into_iter()
        .map(|f| f.fact_id)
        .max()
        .unwrap();
    (id, ep)
}

#[test]
fn links_are_typed_deduplicated_and_kept_in_their_space() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let (works, _) = held(
        &mut e,
        &space,
        "mark works at acme",
        fact("mark", "works_at", "acme"),
    );
    let (based, ep) = held(
        &mut e,
        &space,
        "acme is headquartered in lisbon, near the river",
        fact("acme", "based_in", "lisbon"),
    );
    let (guess, _) = held(
        &mut e,
        &space,
        "mark probably works in lisbon",
        fact("mark", "works_in", "lisbon"),
    );

    let link = e
        .link_facts(&space, guess, works, "derived_from", None, None)
        .unwrap();
    assert_eq!(
        (link.from_fact, link.to_fact, link.kind.as_str()),
        (guess, works, "derived_from")
    );
    let again = e
        .link_facts(&space, guess, works, "derived_from", None, None)
        .unwrap();
    assert_eq!(
        again.link_id, link.link_id,
        "the same link twice is one link"
    );
    let with_quote = e
        .link_facts(
            &space,
            guess,
            based,
            "supports",
            Some(ep),
            Some("headquartered in lisbon"),
        )
        .unwrap();
    assert_eq!(with_quote.source_episode_id, Some(ep));
    assert_eq!(with_quote.quote.as_deref(), Some("headquartered in lisbon"));
    assert!(
        matches!(
            e.link_facts(
                &space,
                guess,
                based,
                "contradicts",
                Some(ep),
                Some("in porto")
            ),
            Err(SconeError::InvalidInput(_))
        ),
        "a quote must sit in the episode it names"
    );

    let mut links = e.fact_links(&space, guess).unwrap();
    links.sort_by_key(|l| l.link_id);
    assert_eq!(
        links.iter().map(|l| l.kind.as_str()).collect::<Vec<_>>(),
        ["derived_from", "supports"]
    );
    assert_eq!(
        e.fact_links(&space, works).unwrap().len(),
        1,
        "a link reads from either end"
    );

    assert!(matches!(
        e.link_facts(&space, guess, guess, "supports", None, None),
        Err(SconeError::InvalidInput(_))
    ));
    assert!(matches!(
        e.link_facts(&space, guess, works, "resembles", None, None),
        Err(SconeError::InvalidInput(_))
    ));
    assert!(matches!(
        e.link_facts(&space, guess, 999_999, "supports", None, None),
        Err(SconeError::NotFound(_))
    ));

    let other = auth::resolve(&mut e, "other", true).unwrap();
    let (elsewhere, _) = held(&mut e, &other, "x is y", fact("x", "is", "y"));
    assert!(matches!(
        e.link_facts(&space, guess, elsewhere, "supports", None, None),
        Err(SconeError::NotFound(_))
    ));
    assert!(
        matches!(e.fact_links(&other, guess), Err(SconeError::NotFound(_))),
        "another space sees nothing"
    );
}

#[test]
fn a_dependency_that_would_close_a_cycle_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let (a, _) = held(&mut e, &space, "a is 1", fact("a", "is", "1"));
    let (b, _) = held(&mut e, &space, "b is 2", fact("b", "is", "2"));
    let (c, _) = held(&mut e, &space, "c is 3", fact("c", "is", "3"));
    e.link_facts(&space, b, c, "derived_from", None, None)
        .unwrap();
    e.link_facts(&space, c, a, "extends", None, None).unwrap();
    assert!(
        matches!(
            e.link_facts(&space, a, b, "derived_from", None, None),
            Err(SconeError::InvalidInput(_))
        ),
        "a -> b -> c -> a"
    );
    assert!(matches!(
        e.link_facts(&space, a, b, "extends", None, None),
        Err(SconeError::InvalidInput(_))
    ));
    e.link_facts(&space, a, b, "contradicts", None, None)
        .unwrap();
    assert_eq!(
        e.fact_links(&space, a).unwrap().len(),
        2,
        "contradicts is not a dependency"
    );
}
