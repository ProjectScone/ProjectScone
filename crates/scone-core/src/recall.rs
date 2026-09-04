//! Hybrid retrieval (spec §7): BM25 + vectors fused by reciprocal-rank
//! fusion, recency-weighted, and re-verified against SQLite truth before
//! anything is returned (memory/bugs.md P-3).

use std::collections::HashMap;

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::error::{Result, SconeError};

const CANDIDATES_PER_GENERATOR: usize = 50;

/// Interrogative and function words that pollute the BM25 leg of a
/// natural-language question ("what did I say about X" must rank on X,
/// not on "what"). The embedding leg keeps the full query — word order
/// and function words carry meaning there.
const QUERY_STOPWORDS: [&str; 33] = [
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your", "it", "is", "was", "were",
    "are", "be", "been", "do", "does", "did", "have", "has", "had", "what", "when", "where",
    "which", "who", "how", "why", "say", "said", "about",
];

fn bm25_query(query: &str) -> String {
    let kept: Vec<&str> = query
        .split_whitespace()
        .filter(|t| {
            !QUERY_STOPWORDS.contains(
                &t.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric()),
            )
        })
        .collect();
    if kept.is_empty() {
        query.to_owned()
    } else {
        kept.join(" ")
    }
}
const RRF_K: f32 = 60.0;
const W_FUSED: f32 = 0.8;
const W_RECENCY: f32 = 0.2;
/// Bonus when the query references a date window and the episode falls in
/// it (E12: temporal questions are time-anchored; measured weakest class).
const W_DATE_MATCH: f32 = 0.4;
const RECENCY_HALF_LIFE_DAYS: f32 = 30.0;
/// Weight of the cross-encoder score when a reranker is attached; the
/// fused+recency score keeps the remainder (v1 blend, benchmarked in
/// memory/benchmarks.md).
const W_RERANK: f32 = 0.7;

#[derive(Debug, Clone)]
pub struct RecallOpts {
    pub limit: usize,
    pub budget_bytes: Option<usize>,
    /// Evaluate fact validity at this instant (ISO-8601). None = now.
    /// Time travel: a past `as_of` serves the then-valid closed facts.
    pub as_of: Option<String>,
    /// Widen each hit with its adjacent chunks. Roughly triples context
    /// for a within-noise accuracy change on the retrieval floor
    /// (measured 2026-08-27) — so it is off by default and enabled by
    /// reader-facing surfaces (ask, MCP recall) where a downstream model
    /// benefits from surrounding context.
    pub expand_neighbors: bool,
    /// Focus retrieval to episodes carrying ALL of these tags (facts
    /// narrow through provenance). Empty = no tag filter.
    pub tags: Vec<String>,
}

impl Default for RecallOpts {
    fn default() -> Self {
        Self {
            limit: 10,
            budget_bytes: None,
            as_of: None,
            expand_neighbors: false,
            tags: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FactItem {
    pub fact_id: i64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub valid_from: String,
    pub valid_until: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct RecallItem {
    pub chunk_id: i64,
    pub episode_id: i64,
    pub text: String,
    /// Rank after fusion, normalized to the best hit in this query. It
    /// says how this item compares to the others, not whether any of
    /// them belong: the top hit scores ~1.0 even for a query nothing
    /// in the store is about.
    pub score: f32,
    /// Cosine similarity to the query from the vector lane, when that
    /// lane found it. This is the absolute signal: unlike `score`, it
    /// stays low when nothing relevant exists.
    pub similarity: Option<f32>,
    pub source: Option<String>,
    pub created_at: String,
}

impl RecallItem {
    /// Calendar day of `created_at`, for dating a line of recalled
    /// memory without spending tokens on the full timestamp. Readers
    /// cannot order events or answer "when" without this; benchmarks
    /// scored zero on temporal questions when the date was dropped.
    pub fn day(&self) -> &str {
        self.created_at
            .split('T')
            .next()
            .unwrap_or(&self.created_at)
    }
}

#[derive(Debug, Default)]
pub struct ContextPack {
    /// Semantic facts, first-class and budget-first (spec §7).
    pub facts: Vec<FactItem>,
    pub items: Vec<RecallItem>,
    /// Generators that could not contribute, stated loudly (spec §10).
    pub degraded: Vec<String>,
    /// Bytes of chunk text returned in `items` (MISSION.md: token economy
    /// is a product surface, not a benchmark-only number).
    pub returned_bytes: usize,
    /// Total episode bytes stored in the space at query time.
    pub space_bytes: i64,
}

impl ContextPack {
    /// Fraction of the stored corpus NOT sent: 0.99 = 99% saved.
    pub fn context_reduction(&self) -> f64 {
        if self.space_bytes <= 0 {
            return 0.0;
        }
        1.0 - (self.returned_bytes as f64 / self.space_bytes as f64)
    }
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
        // Staged index writes become visible here (flush-on-recall).
        self.flush_indexes()?;
        let mut degraded = Vec::new();
        let date_windows = crate::timeparse::date_windows(query);

        // Fact generator: entity/predicate/object term match, validity
        // evaluated at `as_of` (spec §5 I2/I3 make this a WHERE clause).
        let facts = self.recall_facts(space, query, opts)?;

        // Candidate generation: each generator may degrade, never abort.
        let fts_hits = match self.fts.search(
            space.id() as u64,
            &bm25_query(query),
            CANDIDATES_PER_GENERATOR,
        ) {
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

        // Keep the vector lane's cosine similarity before fusion throws
        // it away. Rank fusion answers "which of these is best", never
        // "is any of this actually about the question", and only the
        // second question can decide whether to inject anything at all.
        let similarity: HashMap<u64, f32> = vec_hits.iter().copied().collect();

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
            // Tag focus (AND semantics): the episode must carry every
            // requested tag (normalized lowercase).
            let normalized_tags: Vec<String> =
                opts.tags.iter().map(|t| t.trim().to_lowercase()).collect();
            let tag_filter = if normalized_tags.is_empty() {
                String::new()
            } else {
                format!(
                    " AND (SELECT count(DISTINCT t.name) FROM episode_tags et
                           JOIN tags t ON t.id = et.tag_id
                           WHERE et.episode_id = e.id AND t.name IN ({})) = {}",
                    normalized_tags
                        .iter()
                        .map(|_| "?")
                        .collect::<Vec<_>>()
                        .join(","),
                    normalized_tags.len()
                )
            };
            let sql = format!(
                "SELECT c.episode_id, c.start_byte, c.end_byte, e.content, e.source,
                        e.created_at,
                        (julianday('now') - julianday(e.created_at)) AS age_days
                 FROM chunks c JOIN episodes e ON e.id = c.episode_id
                 WHERE c.id = ?1 AND e.space_id = ?2{tag_filter}"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            for (chunk_id, fused_score) in &fused {
                let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
                    vec![Box::new(*chunk_id as i64), Box::new(space.id())];
                for tag in &normalized_tags {
                    params.push(Box::new(tag.clone()));
                }
                let row = stmt.query_row(
                    rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
                    |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, Option<String>>(4)?,
                            r.get::<_, String>(5)?,
                            r.get::<_, f64>(6)?,
                        ))
                    },
                );
                let (episode_id, start, end, content, source, created_at, age_days) = match row {
                    Ok(r) => r,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => return Err(SconeError::Db(e)),
                };
                let (start, end) = (start as usize, end as usize);
                let text = content.get(start..end).unwrap_or_default().to_owned();
                let recency = (-(age_days.max(0.0) as f32) / RECENCY_HALF_LIFE_DAYS).exp();
                let date_bonus = if date_windows.iter().any(|w| {
                    created_at.as_str() >= w.start.as_str() && created_at.as_str() <= w.end.as_str()
                }) {
                    W_DATE_MATCH
                } else {
                    0.0
                };
                let score = W_FUSED * (fused_score / max_fused) + W_RECENCY * recency + date_bonus;
                items.push(RecallItem {
                    chunk_id: *chunk_id as i64,
                    episode_id,
                    text,
                    score,
                    similarity: similarity.get(chunk_id).copied(),
                    source,
                    created_at,
                });
            }
        }
        // Cross-encoder pass: rescore every surviving candidate against
        // the query jointly. Rank fusion proposes; the reranker disposes.
        if let Some(reranker) = &self.reranker
            && !items.is_empty()
        {
            let docs: Vec<&str> = items.iter().map(|i| i.text.as_str()).collect();
            match reranker.rerank(query, &docs) {
                Ok(scores) => {
                    let (lo, hi) = scores
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(lo, hi), s| (lo.min(*s), hi.max(*s)));
                    let span = (hi - lo).max(f32::MIN_POSITIVE);
                    for (item, raw) in items.iter_mut().zip(&scores) {
                        let normalized = (raw - lo) / span;
                        item.score = W_RERANK * normalized + (1.0 - W_RERANK) * item.score;
                    }
                }
                Err(e) => degraded.push(format!("reranker: {e}")),
            }
        }

        items.sort_by(|a, b| b.score.total_cmp(&a.score));

        // Episode diversity: one strong episode must not hog the top slots
        // with many of its chunks; multi-evidence questions need distinct
        // sources in the window (stratified data 2026-08-28: all-evidence
        // recall trails any-evidence by 8 points).
        const MAX_CHUNKS_PER_EPISODE: usize = 2;
        let mut per_episode: HashMap<i64, usize> = HashMap::new();
        let mut picked = Vec::with_capacity(opts.limit);
        let mut overflow = Vec::new();
        for item in items {
            let count = per_episode.entry(item.episode_id).or_insert(0);
            if *count < MAX_CHUNKS_PER_EPISODE {
                *count += 1;
                picked.push(item);
            } else {
                overflow.push(item);
            }
            if picked.len() == opts.limit {
                break;
            }
        }
        // Fill any remaining slots from the overflow, best first.
        for item in overflow {
            if picked.len() == opts.limit {
                break;
            }
            picked.push(item);
        }
        let items = picked;

        // Neighbor expansion: answers often live one chunk over. Widen each
        // kept item to its adjacent chunks (contiguous byte spans, so this
        // is one slice), skipping items swallowed by an earlier span.
        let mut expanded: Vec<RecallItem> = Vec::with_capacity(items.len());
        for mut item in items {
            if !opts.expand_neighbors {
                expanded.push(item);
                continue;
            }
            let row = self.conn.query_row(
                "SELECT e.content,
                        (SELECT min(start_byte) FROM chunks n
                         WHERE n.episode_id = c.episode_id AND n.pos BETWEEN c.pos - 1 AND c.pos + 1),
                        (SELECT max(end_byte) FROM chunks n
                         WHERE n.episode_id = c.episode_id AND n.pos BETWEEN c.pos - 1 AND c.pos + 1)
                 FROM chunks c JOIN episodes e ON e.id = c.episode_id
                 WHERE c.id = ?1",
                [item.chunk_id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                },
            );
            if let Ok((content, start, end)) = row {
                let (start, end) = (start as usize, end as usize);
                if let Some(wider) = content.get(start..end) {
                    item.text = wider.to_owned();
                }
            }
            let redundant = expanded
                .iter()
                .any(|kept| kept.episode_id == item.episode_id && kept.text.contains(&item.text));
            if !redundant {
                expanded.push(item);
            }
        }
        let mut items = expanded;

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

        // Budget: facts are dense and land first (spec §7); chunks share
        // what remains under the pinned top-item rule.
        if let Some(budget) = opts.budget_bytes {
            let facts_bytes: usize = facts
                .iter()
                .map(|f| f.subject.len() + f.predicate.len() + f.object.len())
                .sum();
            let chunk_budget = budget.saturating_sub(facts_bytes);
            let mut used = 0usize;
            let mut kept = Vec::new();
            for item in items {
                if !kept.is_empty() && used + item.text.len() > chunk_budget {
                    break;
                }
                used += item.text.len();
                kept.push(item);
            }
            items = kept;
        }

        let returned_bytes = items.iter().map(|i| i.text.len()).sum();
        let space_bytes = self.conn.query_row(
            "SELECT coalesce(sum(length(content)), 0) FROM episodes WHERE space_id = ?1",
            [space.id()],
            |r| r.get(0),
        )?;
        Ok(ContextPack {
            facts,
            items,
            degraded,
            returned_bytes,
            space_bytes,
        })
    }

    fn recall_facts(
        &mut self,
        space: &ScopedSpace,
        query: &str,
        opts: &RecallOpts,
    ) -> Result<Vec<FactItem>> {
        let terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|t| t.len() > 2)
            .map(|t| format!("%{t}%"))
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let as_of = opts
            .as_of
            .clone()
            .unwrap_or_else(|| "now-sentinel".to_owned());
        let mut found: Vec<FactItem> = Vec::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT f.id, en.canonical, f.predicate, f.object, f.confidence,
                        f.valid_from, f.valid_until, f.status
                 FROM facts f
                 JOIN entities en ON en.id = f.subject_entity
                 WHERE f.space_id = ?1
                   AND f.valid_from <= ?2
                   AND (f.valid_until IS NULL OR f.valid_until > ?2)
                   AND (en.canonical LIKE ?3 OR f.predicate LIKE ?3 OR f.object LIKE ?3
                        OR EXISTS (SELECT 1 FROM entity_aliases a
                                   WHERE a.entity_id = f.subject_entity AND a.alias LIKE ?3))
                 ORDER BY f.confidence DESC, f.access_count DESC
                 LIMIT ?4",
            )?;
            let now: String =
                self.conn
                    .query_row("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')", [], |r| {
                        r.get(0)
                    })?;
            let effective = if as_of == "now-sentinel" { now } else { as_of };
            for term in &terms {
                let rows = stmt.query_map(
                    rusqlite::params![space.id(), effective, term, opts.limit as i64],
                    |r| {
                        Ok(FactItem {
                            fact_id: r.get(0)?,
                            subject: r.get(1)?,
                            predicate: r.get(2)?,
                            object: r.get(3)?,
                            confidence: r.get(4)?,
                            valid_from: r.get(5)?,
                            valid_until: r.get(6)?,
                            status: r.get(7)?,
                        })
                    },
                )?;
                for row in rows {
                    let row = row?;
                    if !found.iter().any(|f| f.fact_id == row.fact_id) {
                        found.push(row);
                    }
                }
            }
        }
        found.truncate(opts.limit);
        // Reinforcement: recalled present-time facts strengthen (spec §7).
        // Historical (as_of) browsing does not rewrite the present.
        if opts.as_of.is_none() {
            for f in &found {
                self.conn.execute(
                    "UPDATE facts SET access_count = access_count + 1,
                            last_accessed = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                     WHERE id = ?1",
                    [f.fact_id],
                )?;
            }
        }
        Ok(found)
    }
}
