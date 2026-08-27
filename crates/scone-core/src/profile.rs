//! Two-tier profiles (spec §7 via memory/lessons.md L-11): the predecessor
//! validated the shape — stable identity facts plus recent activity, one
//! cheap call (their "~50ms profiles"; ours is a pair of indexed queries).

use crate::auth::ScopedSpace;
use crate::error::Result;
use crate::{Engine, FactItem};

#[derive(Debug)]
pub struct Profile {
    /// Durable identity: active facts, strongest first.
    pub static_facts: Vec<FactItem>,
    /// Recent activity: newest episode excerpts, newest first.
    pub dynamic: Vec<String>,
}

impl Engine {
    pub fn profile(&mut self, space: &ScopedSpace, limit: usize) -> Result<Profile> {
        let limit = limit.clamp(1, 50) as i64;
        let static_facts = {
            let mut stmt = self.conn.prepare(
                "SELECT f.id, en.canonical, f.predicate, f.object, f.confidence,
                        f.valid_from, f.valid_until, f.status
                 FROM facts f JOIN entities en ON en.id = f.subject_entity
                 WHERE f.space_id = ?1 AND f.status = 'active'
                 ORDER BY f.access_count DESC, f.confidence DESC, f.id
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(rusqlite::params![space.id(), limit], |r| {
                Ok(FactItem {
                    fact_id: r.get(0)?,
                    subject: r.get(1)?,
                    predicate: r.get(2)?,
                    object: r.get(3)?,
                    confidence: r.get(4)?,
                    valid_from: r.get(5)?,
                    valid_until: r.get(6)?,
                    status: r.get(7)?,
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let dynamic = {
            let mut stmt = self.conn.prepare(
                "SELECT substr(content, 1, 200) FROM episodes
                 WHERE space_id = ?1 ORDER BY id DESC LIMIT ?2",
            )?;
            let rows = stmt.query_map(rusqlite::params![space.id(), limit], |r| {
                r.get::<_, String>(0)
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(Profile {
            static_facts,
            dynamic,
        })
    }
}
