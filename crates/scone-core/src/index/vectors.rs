//! Vector index over chunk embeddings (usearch HNSW, cosine).
//!
//! Derived data: rebuildable from the embedding blobs in SQLite. The
//! dimension is pinned at open; a mismatch is a typed error pointing at
//! `doctor --rebuild` (spec §9), never a silent re-index.

use std::path::{Path, PathBuf};

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::error::{Result, SconeError};

const FILE_NAME: &str = "vectors.usearch";

pub struct VectorIndex {
    index: Index,
    path: PathBuf,
    dim: usize,
}

fn ix(e: impl std::fmt::Display) -> SconeError {
    SconeError::Index(format!("vectors: {e}"))
}

fn new_index(dim: usize) -> Result<Index> {
    let options = IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    usearch::new_index(&options).map_err(ix)
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("path", &self.path)
            .field("dim", &self.dim)
            .field("size", &self.index.size())
            .finish()
    }
}

impl VectorIndex {
    /// Open, resetting the stored file on dimension mismatch. Only the
    /// repair path uses this; normal opens refuse mismatches loudly.
    pub fn open_or_reset(dir: &Path, dim: usize) -> Result<Self> {
        match Self::open(dir, dim) {
            Err(SconeError::Index(_)) => {
                std::fs::remove_file(dir.join(FILE_NAME))?;
                Self::open(dir, dim)
            }
            other => other,
        }
    }

    pub fn open(dir: &Path, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(FILE_NAME);
        let index = new_index(dim)?;
        if path.exists() {
            let p = path
                .to_str()
                .ok_or_else(|| ix("index path is not valid UTF-8"))?;
            index.load(p).map_err(ix)?;
            let stored = index.dimensions();
            if stored != dim {
                return Err(SconeError::Index(format!(
                    "vectors: stored dimension {stored} != embedder dimension {dim}; \
                     run `scone doctor --rebuild`"
                )));
            }
        }
        Ok(Self { index, path, dim })
    }

    pub fn add(&mut self, rows: &[(u64, &[f32])]) -> Result<()> {
        let needed = self.index.size() + rows.len();
        if self.index.capacity() < needed {
            self.index.reserve(needed).map_err(ix)?;
        }
        for (key, vector) in rows {
            if vector.len() != self.dim {
                return Err(SconeError::Index(format!(
                    "vectors: vector for key {key} has dimension {} != {}",
                    vector.len(),
                    self.dim
                )));
            }
            self.index.add(*key, vector).map_err(ix)?;
        }
        self.save()
    }

    /// Top-`k` `(chunk_id, cosine_similarity)`, unfiltered by space —
    /// callers post-filter against truth (memory/bugs.md P-3).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if self.index.size() == 0 {
            return Ok(Vec::new());
        }
        let matches = self.index.search(query, k.max(1)).map_err(ix)?;
        Ok(matches
            .keys
            .iter()
            .zip(&matches.distances)
            .map(|(key, dist)| (*key, 1.0 - dist))
            .collect())
    }

    pub fn wipe(&mut self) -> Result<()> {
        self.index = new_index(self.dim)?;
        self.save()
    }

    fn save(&self) -> Result<()> {
        let p = self
            .path
            .to_str()
            .ok_or_else(|| ix("index path is not valid UTF-8"))?;
        self.index.save(p).map_err(ix)
    }
}
