//! Deleting a whole space: the preview says what would go, the deed
//! removes it in order and marks the space deleted so the name cannot be
//! re-created. The same receipt shape as the Python engine
//! (tests/fixtures/space-receipt.json); what this engine cannot hold
//! (tombstones, attachments) is 0 or empty.

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::error::{Result, SconeError};

#[derive(Debug, Clone, PartialEq)]
pub struct SpaceReceipt {
    pub space: String,
    pub episodes: i64,
    pub chunks: i64,
    pub facts: i64,
    pub links: i64,
    pub tombstones: i64,
    pub events: i64,
    pub attachments_released: Vec<String>,
    pub attachments_kept: Vec<String>,
    /// Set once the deed is done; a preview has none.
    pub deleted_at: Option<String>,
}

fn count(conn: &rusqlite::Connection, sql: &str, space_id: i64) -> Result<i64> {
    Ok(conn.query_row(sql, [space_id], |r| r.get(0))?)
}

impl Engine {
    /// When the space was deleted, or None while it lives or never was.
    pub fn space_deleted(&self, name: &str) -> Result<Option<String>> {
        match self.conn.query_row(
            "SELECT deleted_at FROM spaces WHERE name = ?1",
            [name],
            |r| r.get::<_, Option<String>>(0),
        ) {
            Ok(when) => Ok(when),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(other) => Err(SconeError::Db(other)),
        }
    }

    fn space_alive(&self, space: &ScopedSpace) -> Result<()> {
        match self.space_deleted(space.name())? {
            Some(when) => Err(SconeError::NotFound(format!(
                "space {:?} was deleted at {when}",
                space.name()
            ))),
            None => Ok(()),
        }
    }

    /// What deleting the space would take with it, with nothing removed.
    pub fn space_impact(&mut self, space: &ScopedSpace) -> Result<SpaceReceipt> {
        self.space_alive(space)?;
        let id = space.id();
        let c = &self.conn;
        Ok(SpaceReceipt {
            space: space.name().to_owned(),
            episodes: count(c, "SELECT count(*) FROM episodes WHERE space_id = ?1", id)?,
            chunks: count(
                c,
                "SELECT count(*) FROM chunks WHERE episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
                id,
            )?,
            facts: count(c, "SELECT count(*) FROM facts WHERE space_id = ?1", id)?,
            links: count(c, "SELECT count(*) FROM fact_links WHERE space_id = ?1", id)?,
            tombstones: 0,
            events: count(
                c,
                "SELECT count(*) FROM evidence_events WHERE space_id = ?1",
                id,
            )?,
            attachments_released: Vec::new(),
            attachments_kept: Vec::new(),
            deleted_at: None,
        })
    }

    /// Remove everything the space holds, in one transaction, then the
    /// vectors and full-text documents that followed its chunks; mark the
    /// space deleted. Returns the receipt `space_impact` would have shown.
    pub fn delete_space(&mut self, space: &ScopedSpace) -> Result<SpaceReceipt> {
        let mut receipt = self.space_impact(space)?;
        let id = space.id();
        let chunk_ids: Vec<u64> = {
            let mut stmt = self.conn.prepare(
                "SELECT id FROM chunks WHERE episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
            )?;
            let rows = stmt.query_map([id], |r| r.get::<_, i64>(0))?;
            rows.map(|r| r.map(|c| c as u64))
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let tx = self.conn.transaction()?;
        for sql in [
            "DELETE FROM episode_tags WHERE episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
            "DELETE FROM tags WHERE space_id = ?1",
            "DELETE FROM episode_metadata WHERE episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
            "DELETE FROM distill_queue WHERE episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
            "DELETE FROM fact_provenance WHERE fact_id IN (SELECT id FROM facts WHERE space_id = ?1)
               OR episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
            "DELETE FROM fact_links WHERE space_id = ?1",
            "DELETE FROM facts WHERE space_id = ?1",
            "DELETE FROM chunks WHERE episode_id IN (SELECT id FROM episodes WHERE space_id = ?1)",
            "DELETE FROM episodes WHERE space_id = ?1",
            "DELETE FROM evidence_events WHERE space_id = ?1",
        ] {
            tx.execute(sql, [id])?;
        }
        tx.execute(
            "UPDATE spaces SET deleted_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), revision = revision + 1 WHERE id = ?1",
            [id],
        )?;
        let deleted_at: String =
            tx.query_row("SELECT deleted_at FROM spaces WHERE id = ?1", [id], |r| {
                r.get(0)
            })?;
        tx.commit()?;
        // The indexes follow the rows; they sit outside the transaction, and
        // the doctor reports anything a crash between the two leaves behind.
        self.vectors.remove(&chunk_ids)?;
        self.vectors.flush()?;
        self.fts.remove_space(id as u64)?;
        self.fts.commit()?;
        receipt.deleted_at = Some(deleted_at);
        Ok(receipt)
    }
}
