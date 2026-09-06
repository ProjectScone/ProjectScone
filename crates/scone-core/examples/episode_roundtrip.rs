//! Test-only JSONL roundtrip for the shared episode profile, not a migration CLI.
//! Uses temporary storage and a deterministic embedder; no network or model API.
use std::{
    collections::HashMap,
    io::{self, Read},
};

use scone_core::{Engine, auth, embed::HashEmbedder};
use serde_json::Value;
use sha2::{Digest, Sha256};

const EVIDENCE_FIELDS: [&str; 5] = ["type", "kind", "content", "source", "created_at"];

fn in_profile(record: &Value) -> bool {
    let Some(object) = record.as_object() else {
        return false;
    };
    let allowed = [
        "type",
        "kind",
        "content",
        "source",
        "created_at",
        "hash",
        "content_hash",
        "space",
        "episode_id",
        "tags",
        "metadata",
    ];
    let Some(date) = record["created_at"].as_str() else {
        return false;
    };
    // The fixture profile is syntactically canonical UTC milliseconds; the
    // real engines retain their own validation. This is not an RFC parser.
    let canonical = date.len() == 24
        && date.bytes().enumerate().all(|(i, ch)| match i {
            4 | 7 => ch == b'-',
            10 => ch == b'T',
            13 | 16 => ch == b':',
            19 => ch == b'.',
            23 => ch == b'Z',
            _ => ch.is_ascii_digit(),
        });
    object.keys().all(|key| allowed.contains(&key.as_str()))
        && record["type"] == "episode"
        && matches!(record["kind"].as_str(), Some("note" | "file" | "connector"))
        && record["content"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty())
        && object
            .get("source")
            .is_some_and(|v| v.is_null() || v.is_string())
        && object
            .get("tags")
            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty))
        && object
            .get("metadata")
            .is_none_or(|v| v.as_object().is_some_and(|m| m.is_empty()))
        && canonical
}

// CPython str.strip includes these four control characters in addition to
// Unicode White_Space. Trim for identity comparison only, never stored evidence.
fn python_strip(content: &str) -> &str {
    content.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn verified_identity(
    record: &Value,
    source_space: Option<&str>,
) -> Result<&'static str, &'static str> {
    let content = record["content"].as_str().ok_or("missing content")?;
    let source_space = if let Some(declared) = record.get("space") {
        let declared = declared
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("invalid source space")?;
        if source_space.is_some_and(|requested| requested != declared) {
            return Err("declared source space disagrees with --python-source-space");
        }
        Some(declared)
    } else {
        source_space
    };
    if let Some(hash) = record.get("hash") {
        let expected = blake3::hash(content.as_bytes()).to_hex().to_string();
        if hash.as_str() != Some(expected.as_str()) {
            return Err("Rust hash does not match source bytes");
        }
    }
    if let Some(hash) = record.get("content_hash") {
        let space = source_space.ok_or("Python identity requires --python-source-space")?;
        let mut digest = Sha256::new();
        digest.update(space.as_bytes());
        digest.update(b"\0");
        digest.update(python_strip(content).as_bytes());
        let expected = format!("{:x}", digest.finalize());
        if hash.as_str() != Some(expected.as_str()) {
            return Err(
                "identity is not the default for the named source space; custom identities are unsupported",
            );
        }
        return Ok("accepted-verified");
    }
    Ok("accepted-native")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source_space = match args.as_slice() {
        [] => None,
        [flag, space] if flag == "--python-source-space" && !space.is_empty() => {
            Some(space.as_str())
        }
        _ => return Err("usage: episode_roundtrip [--python-source-space SPACE]".into()),
    };
    let mut data = String::new();
    io::stdin().read_to_string(&mut data)?;
    let mut identities = Vec::new();
    let mut evidence_by_identity = HashMap::new();
    let mut batch_space: Option<String> = source_space.map(str::to_owned);
    for (index, line) in data
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let record: Value = serde_json::from_str(line)?;
        if !in_profile(&record) {
            return Err(format!(
                "record {} is outside the episode-only profile; no transfer performed",
                index + 1
            )
            .into());
        }
        let identity = verified_identity(&record, source_space).map_err(|reason| {
            format!(
                "record {}: refused-identity: {reason}; no transfer performed",
                index + 1
            )
        })?;
        if let Some(declared) = record.get("space").and_then(Value::as_str) {
            if batch_space
                .as_deref()
                .is_some_and(|space| space != declared)
            {
                return Err(format!("record {}: refused-space: a transfer cannot merge source spaces; no transfer performed", index + 1).into());
            }
            batch_space = Some(declared.to_owned());
        }
        let content = record["content"].as_str().ok_or("missing content")?;
        let evidence: Vec<Value> = EVIDENCE_FIELDS
            .iter()
            .map(|key| record[*key].clone())
            .collect();
        // Distinct source evidence must not collapse on either engine. This
        // checks the complete input batch, not future writes to another store.
        if let Some(previous) =
            evidence_by_identity.insert(python_strip(content).to_owned(), evidence.clone())
            && previous != evidence
        {
            let differing = EVIDENCE_FIELDS
                .iter()
                .enumerate()
                .filter_map(|(i, field)| (previous[i] != evidence[i]).then_some(*field))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("record {}: refused-collision: conflicting {differing} shares a Python dedup identity; no transfer performed", index + 1).into());
        }
        identities.push(serde_json::json!({"record": index + 1, "identity": identity}));
    }
    let dir = tempfile::tempdir()?;
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64)))?;
    let space = auth::resolve(&mut engine, "transfer", true)?;
    engine.import_jsonl(&space, &data)?;
    let repeat = engine.import_jsonl(&space, &data)?;
    if repeat.episodes != 0 || repeat.facts != 0 || repeat.aliases != 0 {
        return Err("reimport changed the store".into());
    }
    let exported = engine.export_jsonl(&space)?;
    for identity in identities {
        eprintln!("SCONE_EPISODE_REPORT {identity}");
    }
    print!("{exported}");
    Ok(())
}
