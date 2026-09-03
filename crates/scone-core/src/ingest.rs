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

/// Default target chunk size in bytes (~250 tokens).
pub(crate) const CHUNK_TARGET_BYTES: usize = 1000;

/// Episode kinds. Closed on purpose: the column is meant to stay
/// answerable. Anything pulled from an outside service is a
/// "connector" episode; which service it was lives in its tag and its
/// source URL, so adding a provider never widens this list.
const KINDS: [&str; 5] = ["note", "file", "conversation", "observation", "connector"];

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
                let is_pdf = path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
                let text = if is_pdf {
                    #[cfg(feature = "pdf")]
                    {
                        pdf_extract::extract_text(&path).map_err(|e| {
                            SconeError::InvalidInput(format!(
                                "{}: pdf extraction failed: {e}",
                                path.display()
                            ))
                        })?
                    }
                    #[cfg(not(feature = "pdf"))]
                    {
                        return Err(SconeError::InvalidInput(format!(
                            "{}: this build lacks the pdf feature",
                            path.display()
                        )));
                    }
                } else {
                    let bytes = std::fs::read(&path)?;
                    String::from_utf8(bytes).map_err(|_| {
                        SconeError::InvalidInput(format!("{} is not valid UTF-8", path.display()))
                    })?
                };
                ("file", text, Some(path.display().to_string()))
            }
        };
        self.ingest_raw(space, kind, &content, source.as_deref(), None)
    }

    /// Import-grade ingest: explicit kind/source/created_at, same pipeline.
    /// Returns the episode id and whether it was freshly stored. Public for
    /// importers and harnesses that must preserve original timestamps —
    /// temporal scoring is only as real as created_at.
    pub fn import_episode(
        &mut self,
        space: &ScopedSpace,
        kind: &str,
        content: &str,
        source: Option<&str>,
        created_at: Option<&str>,
    ) -> Result<(i64, bool)> {
        match self.ingest_raw(space, kind, content, source, created_at)? {
            IngestOutcome::Ingested { episode_id, .. } => Ok((episode_id, true)),
            IngestOutcome::Deduplicated { episode_id } => Ok((episode_id, false)),
        }
    }

    fn ingest_raw(
        &mut self,
        space: &ScopedSpace,
        kind: &str,
        content: &str,
        source: Option<&str>,
        created_at: Option<&str>,
    ) -> Result<IngestOutcome> {
        if !KINDS.contains(&kind) {
            return Err(SconeError::InvalidInput(format!(
                "kind must be one of {KINDS:?}, got {kind:?}"
            )));
        }
        if content.trim().is_empty() {
            return Err(SconeError::InvalidInput("content is empty".into()));
        }
        self.require_writable()?;

        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let spans = chunk_text(content, self.chunk_target);
        let texts: Vec<&str> = spans.iter().map(|s| &content[s.start..s.end]).collect();
        let embeddings = self.embedder.embed(&texts)?;

        let tx = self.conn.transaction()?;
        let inserted = tx.execute(
            "INSERT INTO episodes (space_id, kind, content, hash, source, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5,
                     COALESCE(?6, strftime('%Y-%m-%dT%H:%M:%fZ','now')))
             ON CONFLICT (space_id, hash) DO NOTHING",
            rusqlite::params![space.id(), kind, content, hash, source, created_at],
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
        // Lane 2 is asynchronous: enqueue for distillation, never block
        // ingest on a model (spec §6).
        tx.execute(
            "INSERT OR IGNORE INTO distill_queue (episode_id) VALUES (?1)",
            [episode_id],
        )?;
        // Pin the embedder identity on first write (spec §9); later opens
        // with a different embedder are refused until doctor --rebuild.
        tx.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('embedder_id', ?1)",
            [self.embedder.id()],
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('embedder_dim', ?1)",
            [self.embedder.dim().to_string()],
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
        match index_result {
            Ok(()) => self.indexes_dirty = true,
            Err(_) => self.set_meta("index_dirty", "1")?,
        }

        Ok(IngestOutcome::Ingested {
            episode_id,
            chunks: spans.len(),
        })
    }
}

/// Result of one directory scan.
#[derive(Debug, Default)]
pub struct ScanReport {
    pub ingested: usize,
    pub deduplicated: usize,
    pub skipped: usize,
}

impl Engine {
    /// Recursively ingest text files under `dir`. Hidden entries and
    /// dependency/build directories are skipped; binary and oversized
    /// files are counted in `skipped`, never silently dropped (spec §10).
    /// Content-hash dedup makes rescans cheap and edits append-only.
    pub fn ingest_directory(
        &mut self,
        space: &ScopedSpace,
        dir: &std::path::Path,
        max_file_bytes: u64,
    ) -> Result<ScanReport> {
        self.ingest_directory_tagged(space, dir, max_file_bytes, &[])
    }

    /// Directory scan with curation tags: every ingested (or re-seen) file
    /// gets the given tags plus its lowercase extension as a source tag.
    pub fn ingest_directory_tagged(
        &mut self,
        space: &ScopedSpace,
        dir: &std::path::Path,
        max_file_bytes: u64,
        tags: &[&str],
    ) -> Result<ScanReport> {
        const SKIP_DIRS: [&str; 4] = ["node_modules", "target", ".git", "__pycache__"];
        let mut report = ScanReport::default();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current)? {
                let entry = entry?;
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with('.') {
                    continue;
                }
                let file_type = entry.file_type()?;
                if file_type.is_dir() {
                    if !SKIP_DIRS.contains(&name.as_ref()) {
                        stack.push(path);
                    }
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                if entry.metadata()?.len() > max_file_bytes {
                    report.skipped += 1;
                    continue;
                }
                let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase());
                match self.ingest(space, IngestInput::File { path }) {
                    Ok(outcome) => {
                        let (episode_id, fresh) = match outcome {
                            IngestOutcome::Ingested { episode_id, .. } => (episode_id, true),
                            IngestOutcome::Deduplicated { episode_id } => (episode_id, false),
                        };
                        let mut all: Vec<&str> = tags.to_vec();
                        if let Some(ext) = extension.as_deref() {
                            all.push(ext);
                        }
                        if !all.is_empty() {
                            self.tag_episode(space, episode_id, &all)?;
                        }
                        if fresh {
                            report.ingested += 1;
                        } else {
                            report.deduplicated += 1;
                        }
                    }
                    Err(SconeError::InvalidInput(_)) => report.skipped += 1,
                    Err(other) => return Err(other),
                }
            }
        }
        Ok(report)
    }
}
