#![allow(clippy::unwrap_used)]
//! Two stores built from identical input must retrieve identically.
//!
//! At temperature 0 an A/B leg that changes nothing should score
//! exactly the same, yet one item in twenty-three moved. Temperature is
//! pinned for reader and judge, so any remaining difference comes from
//! below the model, and retrieval is the obvious suspect: usearch builds
//! an HNSW graph, and graph construction is not obliged to be
//! deterministic. This pins whether it is.
/// How much of the ranking is guaranteed identical across builds. The
/// reader-facing default is five, so five is what must hold.
const STABLE_HEAD: usize = 5;

use scone_core::embed::HashEmbedder;
use scone_core::{Engine, RecallOpts, auth};

fn build(dir: &std::path::Path, notes: &[String]) -> Engine {
    let mut e = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    // Explicit dates. Ingesting would stamp the wall clock, so two
    // stores built seconds apart would carry different recency and
    // differ for a legitimate reason, hiding the question being asked.
    for (i, n) in notes.iter().enumerate() {
        let day = 1 + (i % 28);
        e.import_episode(
            &space,
            "note",
            n,
            None,
            Some(&format!("2024-03-{day:02}T12:00:00.000Z")),
        )
        .unwrap();
    }
    e
}

/// Ignored: it fails about one run in three, and that is the finding
/// rather than a defect in the test. usearch builds a randomized HNSW
/// graph and exposes no seed, so two indexes over identical input are
/// different graphs, and an approximate search over different graphs
/// can return different candidates. Items that never reach the
/// candidate list cannot be rescued by tie-breaking. Run it by hand
/// with `cargo test -- --ignored` when changing the index.
#[test]
#[ignore = "cross-build determinism is not achievable with an unseeded HNSW index"]
fn identical_input_retrieves_identically() {
    // Every note distinct. Near-duplicates are a separate question:
    // an approximate index picks arbitrarily among items it cannot
    // tell apart, and which of fifteen identical notes reaches the
    // candidate list is not something tie-breaking can settle, because
    // the loser never arrives to be compared. What must hold is that
    // distinguishable memories retrieve identically.
    let subjects = [
        "deploy target",
        "billing database",
        "office dog",
        "rate limit",
        "conference talk",
        "office lease",
        "retrieval team",
        "vendor contract",
    ];
    let verbs = [
        "moved to",
        "was replaced by",
        "got renamed to",
        "was audited by",
        "was budgeted for",
        "was escalated to",
        "was migrated to",
        "was retired by",
    ];
    let notes: Vec<String> = (0..120)
        .map(|i| {
            format!(
                "{} {} project {i} in quarter {}",
                subjects[i % subjects.len()],
                verbs[(i / 8) % verbs.len()],
                (i % 4) + 1
            )
        })
        .collect();

    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let mut a = build(dir_a.path(), &notes);
    let mut b = build(dir_b.path(), &notes);
    let space_a = auth::resolve(&mut a, "default", true).unwrap();
    let space_b = auth::resolve(&mut b, "default", true).unwrap();

    for query in [
        "where did the deploy target move",
        "quarterly review with the team",
        "host 42",
    ] {
        let opts = RecallOpts {
            limit: 10,
            ..Default::default()
        };
        let pa = a.recall(&space_a, query, &opts).unwrap();
        let pb = b.recall(&space_b, query, &opts).unwrap();
        let ids_a: Vec<i64> = pa.items.iter().map(|i| i.episode_id).collect();
        let ids_b: Vec<i64> = pb.items.iter().map(|i| i.episode_id).collect();
        // The contract is the head, not the tail. HNSW is approximate
        // and its graph is randomized, so the last few results sit
        // among near-ties and can reorder between builds. What a reader
        // is handed first must not.
        assert_eq!(
            ids_a[..STABLE_HEAD],
            ids_b[..STABLE_HEAD],
            "{query:?}: the top {STABLE_HEAD} differ between identical stores\n  {ids_a:?}\n  {ids_b:?}"
        );
    }
}

/// Ties must break on identity, not on arrival order.
///
/// Tested at the unit that decides it, with hand-built ties, because
/// ties cannot be constructed end to end: identical text deduplicates
/// to one episode at ingest, and different text does not score
/// identically. An earlier version recalled repeatedly and hoped to
/// catch a reorder; it passed with the tie-break removed.
#[test]
fn tied_scores_order_by_identity() {
    use scone_core::{RecallItem, order_items};
    let item = |chunk_id: i64, score: f32| RecallItem {
        chunk_id,
        episode_id: chunk_id,
        text: String::new(),
        score,
        similarity: None,
        source: None,
        created_at: String::new(),
    };
    // Arrive in scrambled order with several exact ties.
    let mut items = vec![
        item(7, 0.5),
        item(3, 0.9),
        item(9, 0.5),
        item(1, 0.9),
        item(4, 0.5),
    ];
    order_items(&mut items);
    let ids: Vec<i64> = items.iter().map(|i| i.chunk_id).collect();
    assert_eq!(ids, vec![1, 3, 4, 7, 9], "best score first, ties by id");
}
