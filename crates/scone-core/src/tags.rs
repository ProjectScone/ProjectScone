//! User-controlled tags (2026-08-28): focused retrieval and source
//! curation. Spaces isolate; tags focus within a space. Facts inherit
//! tags through provenance, so a tag filter narrows both lanes.

use crate::Engine;
use crate::auth::ScopedSpace;
use crate::error::{Result, SconeError};

fn validate(name: &str) -> Result<String> {
    let name = name.trim().to_lowercase();
    if name.is_empty() || name.len() > 64 {
        return Err(SconeError::InvalidInput(
            "tag names must be 1..=64 chars".into(),
        ));
    }
    Ok(name)
}

impl Engine {
    /// Attach tags to an episode (idempotent; names normalized lowercase).
    pub fn tag_episode(
        &mut self,
        space: &ScopedSpace,
        episode_id: i64,
        names: &[&str],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        let owned: i64 = tx.query_row(
            "SELECT count(*) FROM episodes WHERE id = ?1 AND space_id = ?2",
            rusqlite::params![episode_id, space.id()],
            |r| r.get(0),
        )?;
        if owned == 0 {
            return Err(SconeError::NotFound(format!(
                "episode {episode_id} in space {}",
                space.name()
            )));
        }
        for raw in names {
            let name = validate(raw)?;
            tx.execute(
                "INSERT OR IGNORE INTO tags (space_id, name) VALUES (?1, ?2)",
                rusqlite::params![space.id(), name],
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO episode_tags (episode_id, tag_id)
                 SELECT ?1, id FROM tags WHERE space_id = ?2 AND name = ?3",
                rusqlite::params![episode_id, space.id(), name],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Tags of a space with episode counts, most-used first.
    pub fn tags_list(&self, space: &ScopedSpace) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.name, count(et.episode_id)
             FROM tags t LEFT JOIN episode_tags et ON et.tag_id = t.id
             WHERE t.space_id = ?1
             GROUP BY t.id ORDER BY count(et.episode_id) DESC, t.name",
        )?;
        let rows = stmt.query_map([space.id()], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}
