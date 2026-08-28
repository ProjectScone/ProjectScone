//! Cross-encoder reranking (spec §7 refinement, 2026-08-27): reads the
//! query and each candidate together, scoring true relevance — the
//! precision lever bi-encoders can't reach. Opt-in: the model is a
//! sizeable one-time download, so surfaces enable it explicitly.

use crate::error::Result;

pub trait Reranker: Send {
    fn id(&self) -> &str;
    /// Relevance score per document (higher = more relevant). Scores are
    /// model-scale (logits); callers normalize across the candidate set.
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

/// Test double: scores 1.0 for documents containing the needle, else 0.0.
pub struct FakeReranker {
    needle: String,
}

impl FakeReranker {
    pub fn preferring(needle: &str) -> Self {
        Self {
            needle: needle.to_lowercase(),
        }
    }
}

impl Reranker for FakeReranker {
    fn id(&self) -> &str {
        "fake-reranker"
    }

    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        Ok(documents
            .iter()
            .map(|d| {
                if d.to_lowercase().contains(&self.needle) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect())
    }
}

/// Local ONNX cross-encoder (bge-reranker-base), cached like the embedder.
#[cfg(feature = "local-embed")]
pub struct OnnxReranker {
    model: std::cell::RefCell<fastembed::TextRerank>,
}

#[cfg(feature = "local-embed")]
impl OnnxReranker {
    pub fn new(cache_dir: &std::path::Path) -> Result<Self> {
        let options = fastembed::RerankInitOptions::new(fastembed::RerankerModel::BGERerankerBase)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(false);
        let model = fastembed::TextRerank::try_new(options)
            .map_err(|e| crate::SconeError::Embed(format!("reranker: {e}")))?;
        Ok(Self {
            model: std::cell::RefCell::new(model),
        })
    }
}

#[cfg(feature = "local-embed")]
impl Reranker for OnnxReranker {
    fn id(&self) -> &str {
        "bge-reranker-base"
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let results = self
            .model
            .borrow_mut()
            .rerank(query, documents, false, None)
            .map_err(|e| crate::SconeError::Embed(format!("reranker: {e}")))?;
        let mut scores = vec![0.0f32; documents.len()];
        for r in results {
            if let Some(slot) = scores.get_mut(r.index) {
                *slot = r.score;
            }
        }
        Ok(scores)
    }
}
