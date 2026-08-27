use std::path::Path;

use rusqlite::Connection;

use crate::error::Result;

/// Schema v1: SQLite is the single source of truth; every other index is a
/// derived, rebuildable cache (spec §5).
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spaces (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    revision   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY,
    space_id   INTEGER NOT NULL REFERENCES spaces(id),
    kind       TEXT NOT NULL CHECK (kind IN ('note','file','conversation','observation')),
    content    TEXT NOT NULL,
    hash       TEXT NOT NULL,
    source     TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (space_id, hash)
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    pos        INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte   INTEGER NOT NULL,
    embedding  BLOB,
    UNIQUE (episode_id, pos)
);
";

/// Schema v2 (spec §5, M2): the semantic store — temporal facts with
/// provenance (I4), entity resolution, and the lane-2 work queue.
const SCHEMA_V2: &str = "
CREATE TABLE IF NOT EXISTS entities (
    id        INTEGER PRIMARY KEY,
    canonical TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias     TEXT PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id)
);
CREATE TABLE IF NOT EXISTS facts (
    id             INTEGER PRIMARY KEY,
    space_id       INTEGER NOT NULL REFERENCES spaces(id),
    subject_entity INTEGER NOT NULL REFERENCES entities(id),
    predicate      TEXT NOT NULL,
    object         TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.5,
    valid_from     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    valid_until    TEXT,
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active','closed','expired')),
    status_reason  TEXT,
    last_accessed  TEXT,
    access_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS facts_subject_predicate
    ON facts (space_id, subject_entity, predicate, status);
CREATE TABLE IF NOT EXISTS fact_provenance (
    fact_id    INTEGER NOT NULL REFERENCES facts(id),
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    UNIQUE (fact_id, episode_id)
);
CREATE TABLE IF NOT EXISTS distill_queue (
    id         INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL UNIQUE REFERENCES episodes(id),
    state      TEXT NOT NULL DEFAULT 'pending'
               CHECK (state IN ('pending','done','failed')),
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
";

pub(crate) fn open(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.execute_batch(SCHEMA)?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1')",
        [],
    )?;
    let version: String = conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'",
        [],
        |r| r.get(0),
    )?;
    if version.as_str() < "2" {
        conn.execute_batch(SCHEMA_V2)?;
        // Episodes ingested before v2 still deserve distillation.
        conn.execute(
            "INSERT OR IGNORE INTO distill_queue (episode_id)
             SELECT id FROM episodes",
            [],
        )?;
        conn.execute(
            "UPDATE meta SET value = '2' WHERE key = 'schema_version'",
            [],
        )?;
    }
    Ok(conn)
}
