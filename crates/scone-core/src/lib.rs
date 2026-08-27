//! Scone core: a local-first episodic + semantic memory engine.
//!
//! This crate performs no stdout I/O and touches the network only through
//! provider traits. SQLite is the single source of truth (spec §5).

pub mod auth;
pub mod chunker;
mod db;
pub mod embed;
mod error;
mod ingest;

use std::path::{Path, PathBuf};

use rusqlite::Connection;

pub use error::{Result, SconeError};
pub use ingest::{IngestInput, IngestOutcome};

pub struct Engine {
    conn: Connection,
    data_dir: PathBuf,
    embedder: Box<dyn embed::EmbeddingProvider>,
}

impl Engine {
    /// Open (creating if needed) the engine rooted at `data_dir`.
    pub fn open(data_dir: &Path, embedder: Box<dyn embed::EmbeddingProvider>) -> Result<Engine> {
        std::fs::create_dir_all(data_dir)?;
        let conn = db::open(&data_dir.join("scone.db"))?;
        Ok(Engine {
            conn,
            data_dir: data_dir.to_path_buf(),
            embedder,
        })
    }

    pub fn space_revision(&self, space: &auth::ScopedSpace) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT revision FROM spaces WHERE id = ?1",
            [space.id()],
            |r| r.get(0),
        )?)
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
