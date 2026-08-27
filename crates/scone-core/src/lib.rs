//! Scone core: a local-first episodic + semantic memory engine.
//!
//! This crate performs no stdout I/O and touches the network only through
//! provider traits. SQLite is the single source of truth (spec §5).

pub mod auth;
pub mod chunker;
mod db;
pub mod embed;
mod error;
pub mod index;
mod ingest;
mod recall;

use std::path::{Path, PathBuf};

use rusqlite::Connection;

pub use error::{Result, SconeError};
pub use ingest::{IngestInput, IngestOutcome};
pub use recall::{ContextPack, RecallItem, RecallOpts};

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
    fts: index::fts::FtsIndex,
    vectors: index::vectors::VectorIndex,
}

impl Engine {
    /// Open (creating if needed) the engine rooted at `data_dir`.
    pub fn open(data_dir: &Path, embedder: Box<dyn embed::EmbeddingProvider>) -> Result<Engine> {
        std::fs::create_dir_all(data_dir)?;
        let conn = db::open(&data_dir.join("scone.db"))?;
        let fts = index::fts::FtsIndex::open(&data_dir.join("fts"))?;
        let vectors = index::vectors::VectorIndex::open(&data_dir.join("vectors"), embedder.dim())?;
        Ok(Engine {
            conn,
            data_dir: data_dir.to_path_buf(),
            embedder,
            fts,
            vectors,
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
