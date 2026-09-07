//! Typed relations between facts of one space, the contract the Python
//! engine holds: supersession lives in the fact, every other relation is
//! a row here. A link names the fact making the claim (`from_fact`) and
//! the fact it is about (`to_fact`), one of four kinds, and the episode
//! and quote it rests on when there is one.

use crate::auth::ScopedSpace;
use crate::{Engine, Result, SconeError};

/// The relation kinds a link may carry.
pub const LINK_KINDS: [&str; 4] = ["extends", "derived_from", "contradicts", "supports"];
/// The kinds that make one fact depend on another; a cycle among them is refused.
const DEPENDENCY_KINDS: [&str; 2] = ["extends", "derived_from"];

/// One stored relation between two facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactLinkItem {
    pub link_id: i64,
    pub from_fact: i64,
    pub to_fact: i64,
    pub kind: String,
    pub created_at: String,
    pub source_episode_id: Option<i64>,
    pub quote: Option<String>,
}

impl Engine {
    /// Relate two facts of `space`. The same (from, to, kind) stored again
    /// returns the stored link unchanged; a quote must sit in the episode
    /// it names; a fact cannot be linked to itself; a dependency that
    /// would close a cycle is refused. Nothing is written until every
    /// check has passed.
    pub fn link_facts(
        &mut self,
        space: &ScopedSpace,
        from_fact: i64,
        to_fact: i64,
        kind: &str,
        source_episode_id: Option<i64>,
        quote: Option<&str>,
    ) -> Result<FactLinkItem> {
        if !LINK_KINDS.contains(&kind) {
            return Err(SconeError::InvalidInput(format!(
                "link kind must be one of {LINK_KINDS:?}, got {kind:?}"
            )));
        }
        if from_fact == to_fact {
            return Err(SconeError::InvalidInput(
                "a fact cannot be linked to itself".into(),
            ));
        }
        for fact_id in [from_fact, to_fact] {
            self.fact_in_space(space, fact_id)?;
        }
        if let Some(text) = quote {
            if text.trim().is_empty() || text.chars().count() > 2000 {
                return Err(SconeError::InvalidInput(
                    "quote must be 1..2000 characters when given".into(),
                ));
            }
            let episode_id = source_episode_id.ok_or_else(|| {
                SconeError::InvalidInput(
                    "a quote needs a source_episode_id to be checked against".into(),
                )
            })?;
            let content: String = self
                .conn
                .query_row(
                    "SELECT content FROM episodes WHERE id = ?1 AND space_id = ?2",
                    rusqlite::params![episode_id, space.id()],
                    |r| r.get(0),
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => SconeError::NotFound(format!(
                        "source episode {episode_id} in space {}",
                        space.name()
                    )),
                    other => SconeError::Db(other),
                })?;
            if !content.contains(text) {
                return Err(SconeError::InvalidInput(format!(
                    "quote is not a substring of episode {episode_id}; the link was not stored"
                )));
            }
        }
        if let Some(existing) = self.find_link(space, from_fact, to_fact, kind)? {
            return Ok(existing);
        }
        if DEPENDENCY_KINDS.contains(&kind) && self.depends_on(space, to_fact, from_fact)? {
            return Err(SconeError::InvalidInput(format!(
                "fact {from_fact} cannot {} fact {to_fact}: that would close a dependency cycle",
                kind.replace('_', " ")
            )));
        }
        self.conn.execute(
            "INSERT OR IGNORE INTO fact_links (space_id, from_fact, to_fact, kind, source_episode_id, quote)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![space.id(), from_fact, to_fact, kind, source_episode_id, quote],
        )?;
        self.find_link(space, from_fact, to_fact, kind)?
            .ok_or_else(|| SconeError::Index("fact link was not stored".into()))
    }

    /// Every link naming the fact at either end, oldest first.
    pub fn fact_links(&self, space: &ScopedSpace, fact_id: i64) -> Result<Vec<FactLinkItem>> {
        self.fact_in_space(space, fact_id)?;
        let mut stmt = self.conn.prepare(
            "SELECT id, from_fact, to_fact, kind, created_at, source_episode_id, quote
             FROM fact_links WHERE space_id = ?1 AND (from_fact = ?2 OR to_fact = ?2)
             ORDER BY id",
        )?;
        let rows = stmt.query_map(rusqlite::params![space.id(), fact_id], link_row)?;
        rows.map(|r| r.map_err(SconeError::Db)).collect()
    }

    fn fact_in_space(&self, space: &ScopedSpace, fact_id: i64) -> Result<()> {
        let found: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM facts WHERE id = ?1 AND space_id = ?2",
                rusqlite::params![fact_id, space.id()],
                |r| r.get(0),
            )
            .ok();
        if found.is_none() {
            return Err(SconeError::NotFound(format!(
                "fact {fact_id} in space {}",
                space.name()
            )));
        }
        Ok(())
    }

    fn find_link(
        &self,
        space: &ScopedSpace,
        from_fact: i64,
        to_fact: i64,
        kind: &str,
    ) -> Result<Option<FactLinkItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, from_fact, to_fact, kind, created_at, source_episode_id, quote
             FROM fact_links WHERE space_id = ?1 AND from_fact = ?2 AND to_fact = ?3 AND kind = ?4",
        )?;
        let mut rows = stmt.query_map(
            rusqlite::params![space.id(), from_fact, to_fact, kind],
            link_row,
        )?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Whether `start` reaches `target` along dependency links.
    fn depends_on(&self, space: &ScopedSpace, start: i64, target: i64) -> Result<bool> {
        let mut seen = std::collections::HashSet::from([start]);
        let mut frontier = vec![start];
        while let Some(fact_id) = frontier.pop() {
            for link in self.fact_links(space, fact_id)? {
                if link.from_fact != fact_id
                    || !DEPENDENCY_KINDS.contains(&link.kind.as_str())
                    || seen.contains(&link.to_fact)
                {
                    continue;
                }
                if link.to_fact == target {
                    return Ok(true);
                }
                seen.insert(link.to_fact);
                frontier.push(link.to_fact);
            }
            if seen.len() > 10_000 {
                return Err(SconeError::InvalidInput(
                    "dependency chain too long to check".into(),
                ));
            }
        }
        Ok(false)
    }
}

fn link_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<FactLinkItem> {
    Ok(FactLinkItem {
        link_id: r.get(0)?,
        from_fact: r.get(1)?,
        to_fact: r.get(2)?,
        kind: r.get(3)?,
        created_at: r.get(4)?,
        source_episode_id: r.get(5)?,
        quote: r.get(6)?,
    })
}
