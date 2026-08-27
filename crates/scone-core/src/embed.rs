//! Embedding providers (spec §9).
//!
//! The engine never talks to a model directly — only through this trait, so
//! local ONNX, remote endpoints, and the deterministic test embedder are
//! interchangeable, and the engine works offline by construction.

use crate::error::Result;

pub trait EmbeddingProvider {
    /// Stable identity, pinned into the index metadata; changing providers
    /// requires `doctor --rebuild` (spec §9).
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Deterministic, model-free embedder: hashed bag-of-words, L2-normalized.
///
/// Exists so tests and degraded mode need no model download; not a semantic
/// embedder.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }
}

impl EmbeddingProvider for HashEmbedder {
    fn id(&self) -> &str {
        "hash-v1"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; self.dim];
                for token in text.to_lowercase().split_whitespace() {
                    // FNV-1a over the token bytes.
                    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                    for b in token.as_bytes() {
                        h ^= u64::from(*b);
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                    let bucket = (h % self.dim as u64) as usize;
                    // Second hash bit decides sign, reducing bucket bias.
                    let sign = if h & (1 << 63) == 0 { 1.0 } else { -1.0 };
                    v[bucket] += sign;
                }
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect())
    }
}
