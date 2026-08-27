#![allow(clippy::unwrap_used)]
use scone_core::embed::{EmbeddingProvider, HashEmbedder};

#[test]
fn hash_embedder_is_deterministic_and_normalized() {
    let e = HashEmbedder::new(64);
    assert_eq!(e.dim(), 64);
    assert_eq!(e.id(), "hash-v1");
    let a = e.embed(&["the quick brown fox"]).unwrap();
    let b = e.embed(&["the quick brown fox"]).unwrap();
    assert_eq!(a, b);
    let c = e.embed(&["completely different words"]).unwrap();
    assert_ne!(a[0], c[0]);
    let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
    assert_eq!(a[0].len(), 64);
}

#[test]
fn empty_text_embeds_to_zero_vector_without_nan() {
    let e = HashEmbedder::new(16);
    let v = e.embed(&[""]).unwrap();
    assert!(v[0].iter().all(|x| *x == 0.0));
}

#[test]
fn batch_embeds_each_text() {
    let e = HashEmbedder::new(32);
    let v = e.embed(&["one", "two", "three"]).unwrap();
    assert_eq!(v.len(), 3);
}
