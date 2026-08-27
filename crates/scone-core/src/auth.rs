//! The single authorization chokepoint (spec invariant I5).
//!
//! Every core read/write API requires a [`ScopedSpace`], and this module is
//! the only place one can be constructed. Access rules live here once, so
//! surfaces cannot re-implement (and diverge on) them — the failure mode the
//! predecessor shipped (memory/bugs.md P-1).

use crate::Engine;
use crate::error::{Result, SconeError};

/// Proof of access to one space. Construction is private to this module.
#[derive(Debug, Clone)]
pub struct ScopedSpace {
    id: i64,
    name: String,
}

impl ScopedSpace {
    pub fn name(&self) -> &str {
        &self.name
    }

    // Consumed by ingest/recall from Task 5 on; scaffolding until then.
    #[allow(dead_code)]
    pub(crate) fn id(&self) -> i64 {
        self.id
    }
}

/// Resolve a space by name, optionally creating it.
///
/// Names are bounded at the surface (memory/bugs.md P-4): 1..=64 chars of
/// `[a-z0-9-_]`.
pub fn resolve(engine: &mut Engine, name: &str, create: bool) -> Result<ScopedSpace> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return Err(SconeError::InvalidInput(format!(
            "space name must be 1..=64 chars of [a-z0-9-_], got {name:?}"
        )));
    }
    let conn = engine.conn_mut();
    if create {
        conn.execute("INSERT OR IGNORE INTO spaces (name) VALUES (?1)", [name])?;
    }
    let id = conn
        .query_row("SELECT id FROM spaces WHERE name = ?1", [name], |r| {
            r.get(0)
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => SconeError::NotFound(format!("space {name:?}")),
            other => SconeError::Db(other),
        })?;
    Ok(ScopedSpace {
        id,
        name: name.to_owned(),
    })
}
