//! Lane 2 fact application (spec §6): entity resolution, provenance,
//! contradiction closure.
//!
//! Invariants enforced here: I2 (no two active facts share subject +
//! predicate), I3 (contradiction closes the old interval, deletes nothing),
//! I4 (every fact carries provenance). The predecessor modeled this as
//! version chains over prose (memory/rationales.md R-3); structure makes
//! contradiction a keyed lookup instead of a prose comparison.

use rusqlite::Transaction;

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::error::{Result, SconeError};
use crate::llm::ExtractedFact;

#[derive(Debug, Default, PartialEq)]
pub struct DistillReport {
    pub processed: usize,
    pub facts_added: usize,
    pub facts_closed: usize,
    pub failed: usize,
}

#[derive(Debug, Default, PartialEq)]
pub struct ApplyReport {
    pub added: usize,
    pub closed: usize,
    pub deduplicated: usize,
}

fn canonicalize(name: &str) -> String {
    name.trim().to_lowercase()
}

fn resolve_entity(tx: &Transaction, name: &str) -> Result<i64> {
    let canonical = canonicalize(name);
    if canonical.is_empty() {
        return Err(SconeError::InvalidInput("empty entity name".into()));
    }
    if let Some(id) = tx
        .query_row(
            "SELECT entity_id FROM entity_aliases WHERE alias = ?1",
            [&canonical],
            |r| r.get::<_, i64>(0),
        )
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(SconeError::Db(other)),
        })?
    {
        return Ok(id);
    }
    tx.execute(
        "INSERT OR IGNORE INTO entities (canonical) VALUES (?1)",
        [&canonical],
    )?;
    Ok(tx.query_row(
        "SELECT id FROM entities WHERE canonical = ?1",
        [&canonical],
        |r| r.get(0),
    )?)
}

impl Engine {
    /// Drain up to `limit` pending episodes of this space through the LLM.
    ///
    /// Failures are recorded on the queue row (attempts, last_error) and
    /// never delete anything (memory/bugs.md P-2); after 3 attempts the row
    /// parks as `failed` and stops being retried implicitly.
    pub fn distill(&mut self, space: &ScopedSpace, limit: usize) -> Result<DistillReport> {
        if self.llm.is_none() {
            return Err(SconeError::Llm(
                "no LLM configured — semantic lane paused; set [llm] in config.toml                  or pass --llm (episodic search is unaffected)"
                    .into(),
            ));
        }
        let pending: Vec<(i64, i64, String)> = {
            let mut stmt = self.conn.prepare(
                "SELECT q.id, q.episode_id, e.content
                 FROM distill_queue q JOIN episodes e ON e.id = q.episode_id
                 WHERE q.state = 'pending' AND e.space_id = ?1
                 ORDER BY q.id LIMIT ?2",
            )?;
            let rows = stmt.query_map(rusqlite::params![space.id(), limit as i64], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut report = DistillReport::default();
        for (queue_id, episode_id, content) in pending {
            let extraction = match &self.llm {
                Some(llm) => llm.extract_facts(&content),
                None => unreachable!("checked above"),
            };
            match extraction {
                Ok(facts) => {
                    let applied = self.apply_facts(space, episode_id, &facts)?;
                    self.conn.execute(
                        "UPDATE distill_queue SET state = 'done', last_error = NULL
                         WHERE id = ?1",
                        [queue_id],
                    )?;
                    report.processed += 1;
                    report.facts_added += applied.added;
                    report.facts_closed += applied.closed;
                }
                Err(e) => {
                    self.conn.execute(
                        "UPDATE distill_queue SET attempts = attempts + 1,
                                last_error = ?1,
                                state = CASE WHEN attempts + 1 >= 3
                                             THEN 'failed' ELSE 'pending' END
                         WHERE id = ?2",
                        rusqlite::params![e.to_string(), queue_id],
                    )?;
                    report.failed += 1;
                }
            }
        }
        Ok(report)
    }

    /// Register `alias` as another name for `canonical` (both canonicalized).
    pub fn add_entity_alias(&mut self, alias: &str, canonical: &str) -> Result<()> {
        let tx = self.conn.transaction()?;
        let entity_id = resolve_entity(&tx, canonical)?;
        tx.execute(
            "INSERT OR REPLACE INTO entity_aliases (alias, entity_id) VALUES (?1, ?2)",
            rusqlite::params![canonicalize(alias), entity_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Apply extracted facts from one episode, in one transaction.
    pub fn apply_facts(
        &mut self,
        space: &ScopedSpace,
        episode_id: i64,
        facts: &[ExtractedFact],
    ) -> Result<ApplyReport> {
        let mut report = ApplyReport::default();
        let tx = self.conn.transaction()?;
        for fact in facts {
            let subject = resolve_entity(&tx, &fact.subject)?;
            let predicate = fact.predicate.trim().to_lowercase();
            let object = fact.object.trim().to_owned();
            if predicate.is_empty() || object.is_empty() {
                return Err(SconeError::InvalidInput(
                    "fact predicate/object must be non-empty".into(),
                ));
            }

            // Exact restatement: strengthen, never duplicate (bugs.md P-5).
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT id FROM facts
                     WHERE space_id = ?1 AND subject_entity = ?2 AND predicate = ?3
                       AND object = ?4 AND status = 'active'",
                    rusqlite::params![space.id(), subject, predicate, object],
                    |r| r.get(0),
                )
                .map(Some)
                .or_else(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => Ok(None),
                    other => Err(SconeError::Db(other)),
                })?;
            if let Some(fact_id) = existing {
                tx.execute(
                    "UPDATE facts SET confidence = max(confidence, ?1) WHERE id = ?2",
                    rusqlite::params![fact.confidence, fact_id],
                )?;
                tx.execute(
                    "INSERT OR IGNORE INTO fact_provenance (fact_id, episode_id) VALUES (?1, ?2)",
                    rusqlite::params![fact_id, episode_id],
                )?;
                report.deduplicated += 1;
                continue;
            }

            tx.execute(
                "INSERT INTO facts (space_id, subject_entity, predicate, object, confidence)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![space.id(), subject, predicate, object, fact.confidence],
            )?;
            let new_id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO fact_provenance (fact_id, episode_id) VALUES (?1, ?2)",
                rusqlite::params![new_id, episode_id],
            )?;
            report.added += 1;

            // Contradiction: same subject+predicate, different object →
            // close the old interval, keep the history (I2/I3).
            let closed = tx.execute(
                "UPDATE facts SET status = 'closed',
                        valid_until = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                        status_reason = 'superseded by fact ' || ?1
                 WHERE space_id = ?2 AND subject_entity = ?3 AND predicate = ?4
                   AND status = 'active' AND id != ?1",
                rusqlite::params![new_id, space.id(), subject, predicate],
            )?;
            report.closed += closed;
        }
        tx.commit()?;
        Ok(report)
    }
}

/// One provenance link: which episode taught us a fact.
#[derive(Debug)]
pub struct ProvenanceItem {
    pub episode_id: i64,
    pub kind: String,
    pub source: Option<String>,
    pub created_at: String,
}

impl Engine {
    /// Facts of a space; active only unless `all` (closed/expired included).
    pub fn facts_list(&self, space: &ScopedSpace, all: bool) -> Result<Vec<crate::FactItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, en.canonical, f.predicate, f.object, f.confidence,
                    f.valid_from, f.valid_until, f.status, f.status_reason
             FROM facts f JOIN entities en ON en.id = f.subject_entity
             WHERE f.space_id = ?1 AND (?2 OR f.status = 'active')
             ORDER BY f.id",
        )?;
        let rows = stmt.query_map(rusqlite::params![space.id(), all], |r| {
            Ok((
                crate::FactItem {
                    fact_id: r.get(0)?,
                    subject: r.get(1)?,
                    predicate: r.get(2)?,
                    object: r.get(3)?,
                    confidence: r.get(4)?,
                    valid_from: r.get(5)?,
                    valid_until: r.get(6)?,
                    status: r.get(7)?,
                },
                r.get::<_, Option<String>>(8)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (mut item, reason) = row?;
            // Carry the closure reason in status for display surfaces.
            if let Some(reason) = reason {
                item.status = format!("{} ({reason})", item.status);
            }
            out.push(item);
        }
        Ok(out)
    }

    /// The episodes that taught us this fact (invariant I4 guarantees ≥1).
    pub fn facts_why(&self, space: &ScopedSpace, fact_id: i64) -> Result<Vec<ProvenanceItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.kind, e.source, e.created_at
             FROM fact_provenance fp
             JOIN facts f ON f.id = fp.fact_id
             JOIN episodes e ON e.id = fp.episode_id
             WHERE fp.fact_id = ?1 AND f.space_id = ?2
             ORDER BY e.id",
        )?;
        let rows = stmt.query_map(rusqlite::params![fact_id, space.id()], |r| {
            Ok(ProvenanceItem {
                episode_id: r.get(0)?,
                kind: r.get(1)?,
                source: r.get(2)?,
                created_at: r.get(3)?,
            })
        })?;
        let out = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        if out.is_empty() {
            return Err(SconeError::NotFound(format!(
                "fact {fact_id} in space {}",
                space.name()
            )));
        }
        Ok(out)
    }

    /// Close a fact by hand, with a reason (interval close, never delete).
    pub fn facts_close(&mut self, space: &ScopedSpace, fact_id: i64, reason: &str) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE facts SET status = 'closed',
                    valid_until = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    status_reason = ?1
             WHERE id = ?2 AND space_id = ?3 AND status = 'active'",
            rusqlite::params![reason, fact_id, space.id()],
        )?;
        if changed == 0 {
            return Err(SconeError::NotFound(format!(
                "active fact {fact_id} in space {}",
                space.name()
            )));
        }
        Ok(())
    }
}
