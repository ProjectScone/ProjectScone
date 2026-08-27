//! Engine benchmarks (spec §11): recall p50 gate < 50ms on-device.
//!
//! The default run is hermetic (hash embedder — measures chunking, SQLite,
//! tantivy, HNSW, fusion, truth re-verification). Set SCONE_BENCH_ONNX=1
//! to also measure true end-to-end recall including local ONNX query
//! embedding (downloads the model once).

#![allow(clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, RecallOpts, auth};

const CORPUS: usize = 5_000;

const TOPICS: &[&str] = &[
    "rust borrow checker ownership lifetimes",
    "sourdough starter hydration baking schedule",
    "quarterly planning meeting budget review",
    "tantivy lucene bm25 segment merging",
    "kubernetes deployment rollout strategy",
    "espresso grinder burr calibration notes",
    "typescript is overrated for engines",
    "vector index hnsw recall tuning",
];

fn seeded_engine(dim: usize) -> (tempfile::TempDir, Engine, scone_core::auth::ScopedSpace) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(dim))).unwrap();
    let space = auth::resolve(&mut engine, "bench", true).unwrap();
    for i in 0..CORPUS {
        let topic = TOPICS[i % TOPICS.len()];
        engine
            .ingest(
                &space,
                IngestInput::Note {
                    text: format!("note {i}: {topic} — observation number {i} about this."),
                },
            )
            .unwrap();
    }
    (dir, engine, space)
}

fn bench_recall(c: &mut Criterion) {
    let (_dir, mut engine, space) = seeded_engine(384);
    let opts = RecallOpts::default();
    c.bench_function("recall_hash_5k", |b| {
        b.iter(|| {
            let pack = engine
                .recall(&space, "borrow checker ownership", &opts)
                .unwrap();
            assert!(!pack.items.is_empty());
        })
    });
}

fn bench_ingest(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(384))).unwrap();
    let space = auth::resolve(&mut engine, "bench", true).unwrap();
    let mut i = 0usize;
    c.bench_function("ingest_note", |b| {
        b.iter(|| {
            i += 1;
            engine
                .ingest(
                    &space,
                    IngestInput::Note {
                        text: format!("unique ingest bench note {i} with fresh content"),
                    },
                )
                .unwrap()
        })
    });
}

fn bench_recall_onnx(c: &mut Criterion) {
    if std::env::var("SCONE_BENCH_ONNX").is_err() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let embedder = scone_core::embed::OnnxEmbedder::new(&dir.path().join("models")).unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(embedder)).unwrap();
    let space = auth::resolve(&mut engine, "bench", true).unwrap();
    for i in 0..500 {
        let topic = TOPICS[i % TOPICS.len()];
        engine
            .ingest(
                &space,
                IngestInput::Note {
                    text: format!("note {i}: {topic}"),
                },
            )
            .unwrap();
    }
    let opts = RecallOpts::default();
    c.bench_function("recall_onnx_500_end_to_end", |b| {
        b.iter(|| {
            engine
                .recall(&space, "vehicle maintenance service", &opts)
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_recall, bench_ingest, bench_recall_onnx);
criterion_main!(benches);
