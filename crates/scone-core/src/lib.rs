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
mod recall;

use std::path::{Path, PathBuf};

use rusqlite::Connection;

pub use distill::{ApplyReport, DistillReport};
pub use error::{Result, SconeError};
pub use ingest::{IngestInput, IngestOutcome};
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
    pub spaces: Vec<SpaceStatus>,
    pub embedder_id: String,
    pub embedder_dim: usize,
    pub index_dirty: bool,
}

pub struct Engine {
    conn: Connection,
    data_dir: PathBuf,
    embedder: Box<dyn embed::EmbeddingProvider>,
    llm: Option<Box<dyn llm::LlmProvider>>,
    fts: index::fts::FtsIndex,
    vectors: index::vectors::VectorIndex,
}

impl Engine {
    /// Open (creating if needed) the engine rooted at `data_dir`.
    ///
    /// Refuses to open when the store was embedded with a different
    /// provider (spec §9): switching embedders is an explicit
    /// `doctor --rebuild`, never a silent re-index.
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
        Ok(Engine {
            conn,
            data_dir: data_dir.to_path_buf(),
            embedder,
            llm: None,
            fts,
            vectors,
        })
    }

    /// Attach or detach the semantic lane's LLM. `None` pauses lane 2
    /// loudly; the episodic engine is unaffected (spec §9).
    pub fn set_llm(&mut self, llm: Option<Box<dyn llm::LlmProvider>>) {
        self.llm = llm;
    }

    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
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
        Ok(StatusReport {
            spaces,
            embedder_id: self.embedder.id().to_owned(),
            embedder_dim: self.embedder.dim(),
            index_dirty: self.get_meta("index_dirty")?.as_deref() == Some("1"),
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
