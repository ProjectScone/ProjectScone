#![allow(clippy::unwrap_used)]
#![cfg(feature = "local-embed")]
use scone_core::embed::{EmbeddingProvider, OnnxEmbedder};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Needs network on first run (model download); run with
/// `cargo test -p scone-core --test onnx -- --ignored`.
#[test]
#[ignore = "downloads the embedding model"]
fn related_sentences_are_closer_than_unrelated() {
    let dir = tempfile::tempdir().unwrap();
    let e = OnnxEmbedder::new(dir.path()).unwrap();
    assert_eq!(e.dim(), 384);
    let v = e
        .embed(&[
            "the cat sat on the windowsill in the sun",
            "a kitten lounged by the sunny window",
            "quarterly revenue exceeded analyst expectations",
        ])
        .unwrap();
    assert!(cosine(&v[0], &v[1]) > cosine(&v[0], &v[2]));
}
