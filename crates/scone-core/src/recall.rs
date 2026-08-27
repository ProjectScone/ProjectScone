//! Hybrid retrieval (spec §7): BM25 + vectors fused by reciprocal-rank
//! fusion, recency-weighted, and re-verified against SQLite truth before
//! anything is returned (memory/bugs.md P-3).

use std::collections::HashMap;

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::error::{Result, SconeError};

const CANDIDATES_PER_GENERATOR: usize = 50;
const RRF_K: f32 = 60.0;
const W_FUSED: f32 = 0.8;
const W_RECENCY: f32 = 0.2;
const RECENCY_HALF_LIFE_DAYS: f32 = 30.0;

#[derive(Debug, Clone)]
pub struct RecallOpts {
    pub limit: usize,
    pub budget_bytes: Option<usize>,
}

impl Default for RecallOpts {
    fn default() -> Self {
        Self {
            limit: 10,
            budget_bytes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecallItem {
    pub chunk_id: i64,
    pub episode_id: i64,
    pub text: String,
    pub score: f32,
    pub source: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Default)]
pub struct ContextPack {
    pub items: Vec<RecallItem>,
    /// Generators that could not contribute, stated loudly (spec §10).
    pub degraded: Vec<String>,
}

impl Engine {
    pub fn recall(
        &mut self,
        space: &ScopedSpace,
        query: &str,
        opts: &RecallOpts,
    ) -> Result<ContextPack> {
        if query.trim().is_empty() {
            return Err(SconeError::InvalidInput("query is empty".into()));
        }
        let mut degraded = Vec::new();

        // Candidate generation: each generator may degrade, never abort.
        let fts_hits = match self
            .fts
            .search(space.id() as u64, query, CANDIDATES_PER_GENERATOR)
        {
            Ok(hits) => hits,
            Err(SconeError::InvalidInput(msg)) => {
                degraded.push(format!("fts: {msg}"));
                Vec::new()
            }
            Err(other) => return Err(other),
        };
        let vec_hits = {
            let q = self.embedder.embed(&[query])?;
            match q.first() {
                Some(qv) => self.vectors.search(qv, CANDIDATES_PER_GENERATOR)?,
                None => Vec::new(),
            }
        };

        // Reciprocal-rank fusion across both ranked lists.
        let mut fused: HashMap<u64, f32> = HashMap::new();
        for hits in [&fts_hits, &vec_hits] {
            for (rank, (chunk_id, _)) in hits.iter().enumerate() {
                *fused.entry(*chunk_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
            }
        }
        let max_fused = fused
            .values()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(f32::MIN_POSITIVE);

        // Truth re-verification: candidates materialize FROM SQLite or not
        // at all; vector hits are space-filtered here too.
        let mut items = Vec::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT c.episode_id, c.start_byte, c.end_byte, e.content, e.source,
                        e.created_at,
                        (julianday('now') - julianday(e.created_at)) AS age_days
                 FROM chunks c JOIN episodes e ON e.id = c.episode_id
                 WHERE c.id = ?1 AND e.space_id = ?2",
            )?;
            for (chunk_id, fused_score) in &fused {
                let row = stmt.query_row(rusqlite::params![*chunk_id as i64, space.id()], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, String>(5)?,
                        r.get::<_, f64>(6)?,
                    ))
                });
                let (episode_id, start, end, content, source, created_at, age_days) = match row {
                    Ok(r) => r,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => return Err(SconeError::Db(e)),
                };
                let (start, end) = (start as usize, end as usize);
                let text = content.get(start..end).unwrap_or_default().to_owned();
                let recency = (-(age_days.max(0.0) as f32) / RECENCY_HALF_LIFE_DAYS).exp();
                let score = W_FUSED * (fused_score / max_fused) + W_RECENCY * recency;
                items.push(RecallItem {
                    chunk_id: *chunk_id as i64,
                    episode_id,
                    text,
                    score,
                    source,
                    created_at,
                });
            }
        }
        items.sort_by(|a, b| b.score.total_cmp(&a.score));
        items.truncate(opts.limit);

        // Budget rule (pinned): the top item always survives; the budget
        // truncates strictly after it.
        if let Some(budget) = opts.budget_bytes {
            let mut used = 0usize;
            let mut kept = Vec::new();
            for item in items {
                if !kept.is_empty() && used + item.text.len() > budget {
                    break;
                }
                used += item.text.len();
                kept.push(item);
            }
            items = kept;
        }

        Ok(ContextPack { items, degraded })
    }
}
