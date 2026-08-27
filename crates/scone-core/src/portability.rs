//! Export/import (spec §8): your memory is portable, full stop.
//!
//! JSONL, one self-describing record per line. Fact provenance crosses
//! stores via episode content hashes (ids are store-local; hashes are
//! identity), and every importer path is idempotent — re-importing an
//! export is a no-op, never a duplication (memory/bugs.md P-5).

use std::collections::HashMap;

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::error::{Result, SconeError};

#[derive(Debug, Default)]
pub struct ImportReport {
    pub episodes: usize,
    pub deduplicated: usize,
    pub facts: usize,
    pub aliases: usize,
}

impl Engine {
    /// Export one space as JSONL: episodes, entity aliases, then facts
    /// (with full interval history and hash-based provenance).
    pub fn export_jsonl(&self, space: &ScopedSpace) -> Result<String> {
        let mut out = String::new();
        let mut stmt = self.conn.prepare(
            "SELECT kind, content, source, created_at, hash
             FROM episodes WHERE space_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map([space.id()], |r| {
            Ok(serde_json::json!({
                "type": "episode",
                "kind": r.get::<_, String>(0)?,
                "content": r.get::<_, String>(1)?,
                "source": r.get::<_, Option<String>>(2)?,
                "created_at": r.get::<_, String>(3)?,
                "hash": r.get::<_, String>(4)?,
            }))
        })?;
        for row in rows {
            out.push_str(&row?.to_string());
            out.push('\n');
        }
        let mut stmt = self.conn.prepare(
            "SELECT a.alias, en.canonical FROM entity_aliases a
             JOIN entities en ON en.id = a.entity_id ORDER BY a.alias",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(serde_json::json!({
                "type": "alias",
                "alias": r.get::<_, String>(0)?,
                "canonical": r.get::<_, String>(1)?,
            }))
        })?;
        for row in rows {
            out.push_str(&row?.to_string());
            out.push('\n');
        }
        let mut stmt = self.conn.prepare(
            "SELECT en.canonical, f.predicate, f.object, f.confidence, f.valid_from,
                    f.valid_until, f.status, f.status_reason,
                    (SELECT json_group_array(e.hash) FROM fact_provenance fp
                     JOIN episodes e ON e.id = fp.episode_id WHERE fp.fact_id = f.id)
             FROM facts f JOIN entities en ON en.id = f.subject_entity
             WHERE f.space_id = ?1 ORDER BY f.id",
        )?;
        let rows = stmt.query_map([space.id()], |r| {
            let provenance: String = r.get(8)?;
            Ok(serde_json::json!({
                "type": "fact",
                "subject": r.get::<_, String>(0)?,
                "predicate": r.get::<_, String>(1)?,
                "object": r.get::<_, String>(2)?,
                "confidence": r.get::<_, f64>(3)?,
                "valid_from": r.get::<_, String>(4)?,
                "valid_until": r.get::<_, Option<String>>(5)?,
                "status": r.get::<_, String>(6)?,
                "status_reason": r.get::<_, Option<String>>(7)?,
                "provenance_hashes": serde_json::from_str::<serde_json::Value>(&provenance)
                    .unwrap_or_else(|_| serde_json::json!([])),
            }))
        })?;
        for row in rows {
            out.push_str(&row?.to_string());
            out.push('\n');
        }
        Ok(out)
    }

    /// Import JSONL produced by [`Engine::export_jsonl`]. Idempotent.
    pub fn import_jsonl(&mut self, space: &ScopedSpace, data: &str) -> Result<ImportReport> {
        let mut report = ImportReport::default();
        let mut records = Vec::new();
        for (n, line) in data.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| SconeError::InvalidInput(format!("line {}: not JSON: {e}", n + 1)))?;
            records.push(value);
        }
        // Pass 1: episodes (dedup via UNIQUE(space, hash) as always).
        let mut hash_to_id: HashMap<String, i64> = HashMap::new();
        for record in &records {
            if record["type"] == "episode" {
                let content = record["content"].as_str().unwrap_or_default();
                let kind = record["kind"].as_str().unwrap_or("note");
                let outcome = self.import_episode(
                    space,
                    kind,
                    content,
                    record["source"].as_str(),
                    record["created_at"].as_str(),
                )?;
                let (id, fresh) = outcome;
                if fresh {
                    report.episodes += 1;
                } else {
                    report.deduplicated += 1;
                }
                hash_to_id.insert(blake3::hash(content.as_bytes()).to_hex().to_string(), id);
            }
        }
        // Pass 2: aliases, then facts with hash-mapped provenance.
        for record in &records {
            match record["type"].as_str() {
                Some("alias") => {
                    let (Some(alias), Some(canonical)) =
                        (record["alias"].as_str(), record["canonical"].as_str())
                    else {
                        continue;
                    };
                    let existed: i64 = self.conn.query_row(
                        "SELECT count(*) FROM entity_aliases WHERE alias = ?1",
                        [alias],
                        |r| r.get(0),
                    )?;
                    self.add_entity_alias(alias, canonical)?;
                    if existed == 0 {
                        report.aliases += 1;
                    }
                }
                Some("fact") => {
                    if self.import_fact(space, record, &hash_to_id)? {
                        report.facts += 1;
                    }
                }
                _ => {}
            }
        }
        Ok(report)
    }
}

impl Engine {
    /// Insert one exported fact if an identical one is not already present.
    fn import_fact(
        &mut self,
        space: &ScopedSpace,
        record: &serde_json::Value,
        hash_to_id: &HashMap<String, i64>,
    ) -> Result<bool> {
        let field = |key: &str| -> Result<&str> {
            record[key]
                .as_str()
                .ok_or_else(|| SconeError::InvalidInput(format!("fact record missing {key}")))
        };
        let subject = field("subject")?;
        let predicate = field("predicate")?;
        let object = field("object")?;
        let valid_from = field("valid_from")?;
        let status = field("status")?;
        if !["active", "closed", "expired"].contains(&status) {
            return Err(SconeError::InvalidInput(format!(
                "fact status {status:?} is not one of active/closed/expired"
            )));
        }
        let confidence = record["confidence"].as_f64().unwrap_or(0.5);
        let valid_until = record["valid_until"].as_str();
        let status_reason = record["status_reason"].as_str();

        // Resolve provenance hashes to local episode ids before writing —
        // a fact without provenance would violate I4.
        let mut episode_ids = Vec::new();
        if let Some(hashes) = record["provenance_hashes"].as_array() {
            for h in hashes.iter().filter_map(|h| h.as_str()) {
                let id = match hash_to_id.get(h) {
                    Some(id) => *id,
                    None => match self.conn.query_row(
                        "SELECT id FROM episodes WHERE space_id = ?1 AND hash = ?2",
                        rusqlite::params![space.id(), h],
                        |r| r.get::<_, i64>(0),
                    ) {
                        Ok(id) => id,
                        Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                        Err(e) => return Err(SconeError::Db(e)),
                    },
                };
                episode_ids.push(id);
            }
        }
        if episode_ids.is_empty() {
            return Err(SconeError::InvalidInput(format!(
                "fact ({subject} {predicate} {object}) references no importable \
                 episode; import episodes first (I4)"
            )));
        }

        let tx = self.conn.transaction()?;
        let entity_id: i64 = {
            let canonical = subject.trim().to_lowercase();
            tx.execute(
                "INSERT OR IGNORE INTO entities (canonical) VALUES (?1)",
                [&canonical],
            )?;
            tx.query_row(
                "SELECT id FROM entities WHERE canonical = ?1",
                [&canonical],
                |r| r.get(0),
            )?
        };
        let exists: i64 = tx.query_row(
            "SELECT count(*) FROM facts
             WHERE space_id = ?1 AND subject_entity = ?2 AND predicate = ?3
               AND object = ?4 AND valid_from = ?5 AND status = ?6",
            rusqlite::params![space.id(), entity_id, predicate, object, valid_from, status],
            |r| r.get(0),
        )?;
        if exists > 0 {
            tx.commit()?;
            return Ok(false);
        }
        tx.execute(
            "INSERT INTO facts (space_id, subject_entity, predicate, object, confidence,
                                valid_from, valid_until, status, status_reason)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                space.id(),
                entity_id,
                predicate,
                object,
                confidence,
                valid_from,
                valid_until,
                status,
                status_reason
            ],
        )?;
        let fact_id = tx.last_insert_rowid();
        for episode_id in episode_ids {
            tx.execute(
                "INSERT OR IGNORE INTO fact_provenance (fact_id, episode_id) VALUES (?1, ?2)",
                rusqlite::params![fact_id, episode_id],
            )?;
        }
        tx.commit()?;
        Ok(true)
    }
}
