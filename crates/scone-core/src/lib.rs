//! Scone core: a local-first episodic + semantic memory engine.
//!
//! This crate performs no stdout I/O and touches the network only through
//! provider traits. SQLite is the single source of truth (spec §5).

pub mod auth;
pub mod chunker;
mod db;
pub mod distill;
pub mod embed;
mod error;
pub mod index;
mod ingest;
pub mod llm;
mod portability;
pub mod profile;
mod recall;
pub mod rerank;
mod tags;
pub mod timeparse;

use std::path::{Path, PathBuf};

use rusqlite::Connection;

pub use distill::{ApplyReport, DistillReport, ProvenanceItem};
pub use error::{Result, SconeError};
pub use ingest::{IngestInput, IngestOutcome, ScanReport};
pub use portability::ImportReport;
pub use profile::Profile;
pub use recall::{ContextPack, FactItem, RecallItem, RecallOpts};

#[derive(Debug)]
pub struct DoctorReport {
    pub episodes: usize,
    pub chunks: usize,
    pub reembedded: usize,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("data_dir", &self.data_dir)
            .field("embedder", &self.embedder.id())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct SpaceStatus {
    pub name: String,
    pub episodes: i64,
    pub chunks: i64,
    pub revision: i64,
}

#[derive(Debug)]
pub struct StatusReport {
    pub read_only: bool,
    pub spaces: Vec<SpaceStatus>,
    pub embedder_id: String,
    pub embedder_dim: usize,
    pub index_dirty: bool,
    pub pending_distill: i64,
    pub failed_distill: i64,
    pub llm_id: Option<String>,
}

pub struct Engine {
    conn: Connection,
    data_dir: PathBuf,
    embedder: Box<dyn embed::EmbeddingProvider>,
    llm: Option<Box<dyn llm::LlmProvider>>,
    reranker: Option<Box<dyn rerank::Reranker>>,
    fts: index::fts::FtsIndex,
    vectors: index::vectors::VectorIndex,
    /// Staged index writes awaiting a flush (flush-on-recall / on drop).
    indexes_dirty: bool,
    /// Chunking granularity for future ingests (tuning knob; benchmarked
    /// sweeps live in memory/benchmarks.md).
    chunk_target: usize,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Best-effort: a failed flush here is recovered by the catch-up
        // reindex on the next open (meta 'indexed_max' high-water mark).
        let _ = self.flush_indexes();
    }
}

impl Engine {
    /// Open (creating if needed) the engine rooted at `data_dir`.
    ///
    /// Refuses to open when the store was embedded with a different
    /// provider (spec §9): switching embedders is an explicit
    /// `doctor --rebuild`, never a silent re-index.
    /// Which embedder this store is pinned to. Callers that judge
    /// absolute similarity need it: a threshold is only meaningful for
    /// the model it was measured against.
    pub fn embedder_id(&self) -> &str {
        self.embedder.id()
    }

    pub fn open(data_dir: &Path, embedder: Box<dyn embed::EmbeddingProvider>) -> Result<Engine> {
        Self::open_inner(data_dir, embedder, false)
    }

    /// Open bypassing the embedder pin, for `doctor --rebuild` only.
    pub fn open_for_repair(
        data_dir: &Path,
        embedder: Box<dyn embed::EmbeddingProvider>,
    ) -> Result<Engine> {
        Self::open_inner(data_dir, embedder, true)
    }

    fn open_inner(
        data_dir: &Path,
        embedder: Box<dyn embed::EmbeddingProvider>,
        repair: bool,
    ) -> Result<Engine> {
        std::fs::create_dir_all(data_dir)?;
        let conn = db::open(&data_dir.join("scone.db"))?;
        if !repair {
            let pinned: Option<String> = match conn.query_row(
                "SELECT value FROM meta WHERE key = 'embedder_id'",
                [],
                |r| r.get(0),
            ) {
                Ok(v) => Some(v),
                Err(rusqlite::Error::QueryReturnedNoRows) => None,
                Err(e) => return Err(SconeError::Db(e)),
            };
            if let Some(pinned) = pinned
                && !(pinned == embedder.id() && Self::pinned_dim(&conn)? == Some(embedder.dim()))
            {
                return Err(SconeError::Index(format!(
                    "store is pinned to embedder {pinned}; opening with {} ({} dims)                      requires `scone doctor --rebuild`",
                    embedder.id(),
                    embedder.dim()
                )));
            }
        }
        let vectors_dir = data_dir.join("vectors");
        let vectors = if repair {
            index::vectors::VectorIndex::open_or_reset(&vectors_dir, embedder.dim())?
        } else {
            index::vectors::VectorIndex::open(&vectors_dir, embedder.dim())?
        };
        let fts = index::fts::FtsIndex::open(&data_dir.join("fts"))?;
        let mut engine = Engine {
            conn,
            data_dir: data_dir.to_path_buf(),
            embedder,
            llm: None,
            reranker: None,
            fts,
            vectors,
            indexes_dirty: false,
            chunk_target: ingest::CHUNK_TARGET_BYTES,
        };
        if engine.fts.writable() {
            engine.catch_up_indexes()?;
        }
        Ok(engine)
    }

    /// Set the chunking granularity (bytes) for future ingests.
    pub fn set_chunk_target(&mut self, bytes: usize) {
        self.chunk_target = bytes.max(64);
    }

    /// True when another scone process holds the index write lock: search
    /// works from committed state; ingest/distill/doctor are refused with
    /// typed errors until the other process exits.
    pub fn is_read_only(&self) -> bool {
        !self.fts.writable()
    }

    fn require_writable(&self) -> Result<()> {
        if self.is_read_only() {
            return Err(SconeError::InvalidInput(
                "this store is read-only: another scone process holds the write lock".into(),
            ));
        }
        Ok(())
    }

    /// Reindex any chunks written after the last successful flush — the
    /// crash-recovery half of lazy flushing. Idempotent (both indexes
    /// upsert), and cheap: only the tail beyond the high-water mark.
    fn catch_up_indexes(&mut self) -> Result<()> {
        let indexed_max: i64 = self
            .get_meta("indexed_max")?
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let chunk_max: i64 =
            self.conn
                .query_row("SELECT coalesce(max(id), 0) FROM chunks", [], |r| r.get(0))?;
        if chunk_max <= indexed_max {
            return Ok(());
        }
        let rows: Vec<(i64, i64, String, Option<Vec<u8>>)> = {
            let mut stmt = self.conn.prepare(
                "SELECT c.id, e.space_id, e.content, c.start_byte, c.end_byte, c.embedding
                 FROM chunks c JOIN episodes e ON e.id = c.episode_id
                 WHERE c.id > ?1 ORDER BY c.id",
            )?;
            let mapped = stmt.query_map([indexed_max], |r| {
                let content: String = r.get(2)?;
                let start: i64 = r.get(3)?;
                let end: i64 = r.get(4)?;
                let text = content
                    .get(start as usize..end as usize)
                    .unwrap_or_default()
                    .to_owned();
                Ok((r.get(0)?, r.get(1)?, text, r.get(5)?))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let dim = self.embedder.dim();
        let fts_rows: Vec<(u64, u64, &str)> = rows
            .iter()
            .map(|(id, space, text, _)| (*id as u64, *space as u64, text.as_str()))
            .collect();
        self.fts.add(&fts_rows)?;
        let mut vec_rows: Vec<(u64, Vec<f32>)> = Vec::new();
        for (id, _, _, blob) in &rows {
            if let Some(b) = blob
                && b.len() == dim * 4
            {
                let v = b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                vec_rows.push((*id as u64, v));
            }
        }
        let borrowed: Vec<(u64, &[f32])> =
            vec_rows.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        self.vectors.add(&borrowed)?;
        self.indexes_dirty = true;
        self.flush_indexes()
    }

    /// Flush staged index writes; advances the high-water mark only after
    /// both indexes are durable.
    pub(crate) fn flush_indexes(&mut self) -> Result<()> {
        if !self.indexes_dirty || self.is_read_only() {
            return Ok(());
        }
        self.fts.commit()?;
        self.vectors.flush()?;
        let chunk_max: i64 =
            self.conn
                .query_row("SELECT coalesce(max(id), 0) FROM chunks", [], |r| r.get(0))?;
        self.set_meta("indexed_max", &chunk_max.to_string())?;
        self.indexes_dirty = false;
        Ok(())
    }

    /// Attach or detach the semantic lane's LLM. `None` pauses lane 2
    /// loudly; the episodic engine is unaffected (spec §9).
    pub fn set_llm(&mut self, llm: Option<Box<dyn llm::LlmProvider>>) {
        self.llm = llm;
    }

    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
    }

    /// Attach or detach a cross-encoder reranker for recall precision.
    pub fn set_reranker(&mut self, reranker: Option<Box<dyn rerank::Reranker>>) {
        self.reranker = reranker;
    }

    /// Answer a question from rendered context via the configured LLM.
    pub fn llm_answer(&self, question: &str, context: &str) -> Result<String> {
        match &self.llm {
            Some(llm) => llm.answer(question, context),
            None => Err(SconeError::Llm("no LLM configured".into())),
        }
    }

    /// Answer with an explicit system prompt (for prompt A/B harnesses).
    pub fn llm_answer_with_system(
        &self,
        system: &str,
        question: &str,
        context: &str,
    ) -> Result<String> {
        match &self.llm {
            Some(llm) => llm.answer_with_system(system, question, context),
            None => Err(SconeError::Llm("no LLM configured".into())),
        }
    }

    fn pinned_dim(conn: &Connection) -> Result<Option<usize>> {
        match conn.query_row(
            "SELECT value FROM meta WHERE key = 'embedder_dim'",
            [],
            |r| r.get::<_, String>(0),
        ) {
            Ok(v) => Ok(v.parse().ok()),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(SconeError::Db(e)),
        }
    }

    /// Rebuild every derived index from SQLite truth, re-embedding chunks
    /// whose stored vectors no longer match the active embedder.
    pub fn doctor_rebuild(&mut self) -> Result<DoctorReport> {
        self.require_writable()?;
        self.fts.wipe()?;
        self.vectors.wipe()?;
        // Slice chunk text in Rust: our offsets are bytes, and SQLite's
        // substr() counts characters — mixing them corrupts multibyte text.
        let rows: Vec<(i64, i64, String, Option<Vec<u8>>)> = {
            let mut stmt = self.conn.prepare(
                "SELECT c.id, e.space_id, e.content, c.start_byte, c.end_byte, c.embedding
                 FROM chunks c JOIN episodes e ON e.id = c.episode_id
                 ORDER BY c.id",
            )?;
            let mapped = stmt.query_map([], |r| {
                let content: String = r.get(2)?;
                let start: i64 = r.get(3)?;
                let end: i64 = r.get(4)?;
                let text = content
                    .get(start as usize..end as usize)
                    .unwrap_or_default()
                    .to_owned();
                Ok((r.get(0)?, r.get(1)?, text, r.get(5)?))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let dim = self.embedder.dim();
        let mut reembedded = 0usize;
        let mut fts_rows = Vec::with_capacity(rows.len());
        let mut vec_data = Vec::with_capacity(rows.len());
        for (chunk_id, space_id, text, blob) in &rows {
            let vector = match blob {
                Some(b) if b.len() == dim * 4 => b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect::<Vec<f32>>(),
                _ => {
                    let embedded = self.embedder.embed(&[text.as_str()])?;
                    let v = embedded
                        .into_iter()
                        .next()
                        .ok_or_else(|| SconeError::Embed("embedder returned no vector".into()))?;
                    let new_blob: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                    self.conn.execute(
                        "UPDATE chunks SET embedding = ?1 WHERE id = ?2",
                        rusqlite::params![new_blob, chunk_id],
                    )?;
                    reembedded += 1;
                    v
                }
            };
            fts_rows.push((*chunk_id as u64, *space_id as u64, text.as_str()));
            vec_data.push((*chunk_id as u64, vector));
        }
        self.fts.add(&fts_rows)?;
        let vec_rows: Vec<(u64, &[f32])> =
            vec_data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        self.vectors.add(&vec_rows)?;
        self.indexes_dirty = true;
        self.flush_indexes()?;
        self.set_meta("embedder_id", self.embedder.id())?;
        self.set_meta("embedder_dim", &dim.to_string())?;
        self.set_meta("index_dirty", "0")?;
        let episodes: i64 = self
            .conn
            .query_row("SELECT count(*) FROM episodes", [], |r| r.get(0))?;
        Ok(DoctorReport {
            episodes: episodes as usize,
            chunks: rows.len(),
            reembedded,
        })
    }

    pub(crate) fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }

    pub(crate) fn get_meta(&self, key: &str) -> Result<Option<String>> {
        match self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(SconeError::Db(e)),
        }
    }

    pub fn space_revision(&self, space: &auth::ScopedSpace) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT revision FROM spaces WHERE id = ?1",
            [space.id()],
            |r| r.get(0),
        )?)
    }

    /// Administrative overview of the whole store (single-user surface;
    /// multi-user servers must scope what they expose of it).
    pub fn status(&self) -> Result<StatusReport> {
        let mut stmt = self.conn.prepare(
            "SELECT s.name, s.revision,
                    (SELECT count(*) FROM episodes e WHERE e.space_id = s.id),
                    (SELECT count(*) FROM chunks c JOIN episodes e ON e.id = c.episode_id
                     WHERE e.space_id = s.id)
             FROM spaces s ORDER BY s.name",
        )?;
        let spaces = stmt
            .query_map([], |r| {
                Ok(SpaceStatus {
                    name: r.get(0)?,
                    revision: r.get(1)?,
                    episodes: r.get(2)?,
                    chunks: r.get(3)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let (pending_distill, failed_distill) = self.conn.query_row(
            "SELECT sum(state = 'pending'), sum(state = 'failed') FROM distill_queue",
            [],
            |r| {
                Ok((
                    r.get::<_, Option<i64>>(0)?.unwrap_or(0),
                    r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                ))
            },
        )?;
        Ok(StatusReport {
            read_only: self.is_read_only(),
            spaces,
            embedder_id: self.embedder.id().to_owned(),
            embedder_dim: self.embedder.dim(),
            index_dirty: self.get_meta("index_dirty")?.as_deref() == Some("1"),
            pending_distill,
            failed_distill,
            llm_id: self.llm.as_ref().map(|l| l.id().to_owned()),
        })
    }

    /// Content and kind of one episode, straight from truth.
    pub fn episode_content(&self, episode_id: i64) -> Result<(String, String)> {
        self.conn
            .query_row(
                "SELECT content, kind FROM episodes WHERE id = ?1",
                [episode_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SconeError::NotFound(format!("episode {episode_id}"))
                }
                other => SconeError::Db(other),
            })
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub(crate) fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    pub fn schema_version(&self) -> Result<i64> {
        let v: String = self.conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )?;
        v.parse()
            .map_err(|_| SconeError::Index("schema_version is not a number".into()))
    }
}
