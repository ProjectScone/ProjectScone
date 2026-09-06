#![allow(clippy::unwrap_used)]
//! Kind, source prefix and created_at bounds narrow recall the way tags
//! do: on the candidates fusion produced. Each filter is checked alone
//! and the prefix is checked against SQL wildcard characters, which must
//! stay literal in a path or a URL.
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, RecallOpts, auth};

fn store(dir: &std::path::Path) -> (Engine, auth::ScopedSpace) {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    // Same topic everywhere so retrieval alone cannot tell them apart.
    let rows: [(&str, &str, Option<&str>, &str); 4] = [
        (
            "note",
            "deploy runbook: rotate the staging keys first",
            None,
            "2024-01-10T09:00:00.000Z",
        ),
        (
            "file",
            "deploy runbook: rotate the staging keys, then restart",
            Some("/ops/runbooks/deploy.md"),
            "2024-02-10T09:00:00.000Z",
        ),
        (
            "file",
            "deploy runbook draft: staging keys rotate weekly",
            Some("/ops/drafts/deploy_v2.md"),
            "2024-03-10T09:00:00.000Z",
        ),
        (
            "conversation",
            "user: where is the deploy runbook for staging keys?",
            Some("session-42"),
            "2024-04-10T09:00:00.000Z",
        ),
    ];
    for (kind, text, source, day) in rows {
        e.import_episode(&space, kind, text, source, Some(day))
            .unwrap();
    }
    (e, space)
}

fn episodes(e: &mut Engine, space: &auth::ScopedSpace, opts: RecallOpts) -> Vec<i64> {
    let mut ids: Vec<i64> = e
        .recall(space, "deploy runbook staging keys", &opts)
        .unwrap()
        .items
        .iter()
        .map(|i| i.episode_id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[test]
fn without_a_filter_every_episode_is_a_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(dir.path());
    assert_eq!(
        episodes(&mut e, &space, RecallOpts::default()),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn kind_keeps_only_that_kind() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(dir.path());
    let opts = RecallOpts {
        kind: Some("file".into()),
        ..Default::default()
    };
    assert_eq!(episodes(&mut e, &space, opts), vec![2, 3]);
}

#[test]
fn source_prefix_is_literal_text_not_a_pattern() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(dir.path());
    let under = |prefix: &str| RecallOpts {
        source_prefix: Some(prefix.into()),
        ..Default::default()
    };
    assert_eq!(episodes(&mut e, &space, under("/ops/runbooks/")), vec![2]);
    assert_eq!(episodes(&mut e, &space, under("/ops/")), vec![2, 3]);
    assert_eq!(episodes(&mut e, &space, under("session-")), vec![4]);
    // "%" and "_" are characters, not wildcards: nothing starts with them.
    assert_eq!(episodes(&mut e, &space, under("%")), Vec::<i64>::new());
    assert_eq!(episodes(&mut e, &space, under("/ops/_")), Vec::<i64>::new());
    // An episode with no source matches no prefix.
    assert_eq!(episodes(&mut e, &space, under("")), vec![2, 3, 4]);
}

#[test]
fn since_and_until_bound_when_it_happened() {
    let dir = tempfile::tempdir().unwrap();
    let (mut e, space) = store(dir.path());
    let since = RecallOpts {
        since: Some("2024-02-10T09:00:00.000Z".into()),
        ..Default::default()
    };
    assert_eq!(
        episodes(&mut e, &space, since),
        vec![2, 3, 4],
        "since is inclusive"
    );
    let until = RecallOpts {
        until: Some("2024-02-10T09:00:00.000Z".into()),
        ..Default::default()
    };
    assert_eq!(
        episodes(&mut e, &space, until),
        vec![1, 2],
        "until is inclusive"
    );
    let window = RecallOpts {
        since: Some("2024-02-01T00:00:00.000Z".into()),
        until: Some("2024-03-31T00:00:00.000Z".into()),
        kind: Some("file".into()),
        ..Default::default()
    };
    assert_eq!(
        episodes(&mut e, &space, window),
        vec![2, 3],
        "filters combine with AND"
    );
}
