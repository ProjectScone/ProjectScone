//! Scone core: a local-first episodic + semantic memory engine.
//!
//! This crate performs no stdout I/O and touches the network only through
//! provider traits. SQLite is the single source of truth (spec §5).

pub mod auth;
pub mod chunker;
mod db;
mod error;

use std::path::{Path, PathBuf};

use rusqlite::Connection;

pub use error::{Result, SconeError};

pub struct Engine {
    conn: Connection,
    data_dir: PathBuf,
}

impl Engine {
    /// Open (creating if needed) the engine rooted at `data_dir`.
    pub fn open(data_dir: &Path) -> Result<Engine> {
        std::fs::create_dir_all(data_dir)?;
        let conn = db::open(&data_dir.join("scone.db"))?;
        Ok(Engine {
            conn,
            data_dir: data_dir.to_path_buf(),
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
