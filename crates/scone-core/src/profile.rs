//! Two-tier profiles (spec §7 via memory/lessons.md L-11): the predecessor
//! validated the shape — stable identity facts plus recent activity, one
//! cheap call (their "~50ms profiles"; ours is a pair of indexed queries).

use crate::auth::ScopedSpace;
use crate::error::Result;
use crate::{Engine, FactItem};

/// One `dynamic` excerpt with the episode it was cut from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecentActivity {
    pub episode_id: i64,
    pub excerpt: String,
    pub created_at: String,
}

#[derive(Debug)]
pub struct Profile {
    /// Durable identity: active facts, strongest first.
    pub static_facts: Vec<FactItem>,
    /// Recent activity: newest episode excerpts, newest first.
    pub dynamic: Vec<String>,
    /// `dynamic` with its evidence: same order, same excerpts, newest
    /// first. The shape both engines share (tests/fixtures/profile-recent.json).
    pub recent: Vec<RecentActivity>,
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
        let recent = {
            let mut stmt = self.conn.prepare(
                "SELECT id, substr(content, 1, 200), created_at FROM episodes
                 WHERE space_id = ?1 ORDER BY id DESC LIMIT ?2",
            )?;
            let rows = stmt.query_map(rusqlite::params![space.id(), limit], |r| {
                Ok(RecentActivity {
                    episode_id: r.get(0)?,
                    excerpt: r.get(1)?,
                    created_at: r.get(2)?,
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let dynamic = recent.iter().map(|r| r.excerpt.clone()).collect();
        Ok(Profile {
            static_facts,
            dynamic,
            recent,
        })
    }
}
