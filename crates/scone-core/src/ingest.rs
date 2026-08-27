//! Lane 1: episodic ingest (spec §6).
//!
//! Synchronous and offline-complete. One SQLite transaction is the unit of
//! truth: episode, chunks, and embeddings commit together or not at all.
//! Failures are typed values; nothing is ever deleted to handle one
//! (memory/bugs.md P-2). Duplicates are a first-class outcome detected by
//! UNIQUE(space_id, hash), never by matching error strings (P-5).

use std::path::PathBuf;

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::chunker::chunk_text;
use crate::error::{Result, SconeError};

/// Target chunk size in bytes (~250 tokens).
pub(crate) const CHUNK_TARGET_BYTES: usize = 1000;

#[derive(Debug)]
pub enum IngestInput {
    Note { text: String },
    File { path: PathBuf },
}

#[derive(Debug)]
pub enum IngestOutcome {
    Ingested { episode_id: i64, chunks: usize },
    Deduplicated { episode_id: i64 },
}

impl Engine {
    pub fn ingest(&mut self, space: &ScopedSpace, input: IngestInput) -> Result<IngestOutcome> {
        let (kind, content, source) = match input {
            IngestInput::Note { text } => ("note", text, None),
            IngestInput::File { path } => {
                let bytes = std::fs::read(&path)?;
                let text = String::from_utf8(bytes).map_err(|_| {
                    SconeError::InvalidInput(format!("{} is not valid UTF-8", path.display()))
                })?;
                ("file", text, Some(path.display().to_string()))
            }
        };
        if content.trim().is_empty() {
            return Err(SconeError::InvalidInput("content is empty".into()));
        }

        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let spans = chunk_text(&content, CHUNK_TARGET_BYTES);
        let texts: Vec<&str> = spans.iter().map(|s| &content[s.start..s.end]).collect();
        let embeddings = self.embedder.embed(&texts)?;

        let tx = self.conn.transaction()?;
        let inserted = tx.execute(
            "INSERT INTO episodes (space_id, kind, content, hash, source)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (space_id, hash) DO NOTHING",
            rusqlite::params![space.id(), kind, content, hash, source],
        )?;
        if inserted == 0 {
            let episode_id = tx.query_row(
                "SELECT id FROM episodes WHERE space_id = ?1 AND hash = ?2",
                rusqlite::params![space.id(), hash],
                |r| r.get(0),
            )?;
            // Nothing was written; the revision must not move (bugs.md P-8).
            return Ok(IngestOutcome::Deduplicated { episode_id });
        }
        let episode_id = tx.last_insert_rowid();
        let mut chunk_ids = Vec::with_capacity(spans.len());
        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks (episode_id, pos, start_byte, end_byte, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (pos, (span, emb)) in spans.iter().zip(&embeddings).enumerate() {
                let blob: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
                stmt.execute(rusqlite::params![
                    episode_id,
                    pos as i64,
                    span.start as i64,
                    span.end as i64,
                    blob
                ])?;
                chunk_ids.push(tx.last_insert_rowid());
            }
        }
        tx.execute(
            "UPDATE spaces SET revision = revision + 1 WHERE id = ?1",
            [space.id()],
        )?;
        tx.commit()?;

        // Feed the derived indexes after truth commits. An index failure
        // never fails the ingest: it marks the indexes dirty for
        // `doctor --rebuild` and the status surface says so (spec §10).
        let fts_rows: Vec<(u64, u64, &str)> = chunk_ids
            .iter()
            .zip(&texts)
            .map(|(id, text)| (*id as u64, space.id() as u64, *text))
            .collect();
        let vec_rows: Vec<(u64, &[f32])> = chunk_ids
            .iter()
            .zip(&embeddings)
            .map(|(id, emb)| (*id as u64, emb.as_slice()))
            .collect();
        let index_result = self
            .fts
            .add(&fts_rows)
            .and_then(|()| self.vectors.add(&vec_rows));
        if index_result.is_err() {
            self.set_meta("index_dirty", "1")?;
        }

        Ok(IngestOutcome::Ingested {
            episode_id,
            chunks: spans.len(),
        })
    }
}
