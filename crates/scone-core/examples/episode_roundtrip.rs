//! Test-only JSONL roundtrip for the shared episode profile, not a migration CLI.
//! Uses temporary storage and a deterministic embedder; no network or model API.
use std::io::{self, Read};

use scone_core::{Engine, auth, embed::HashEmbedder};
use serde_json::Value;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = String::new();
    io::stdin().read_to_string(&mut data)?;
    for line in data.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        if !in_profile(&record) {
            return Err("record is outside the episode-only profile; no transfer performed".into());
        }
    }
    let dir = tempfile::tempdir()?;
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64)))?;
    let space = auth::resolve(&mut engine, "transfer", true)?;
    engine.import_jsonl(&space, &data)?;
    let repeat = engine.import_jsonl(&space, &data)?;
    if repeat.episodes != 0 || repeat.facts != 0 || repeat.aliases != 0 {
        return Err("reimport changed the store".into());
    }
    print!("{}", engine.export_jsonl(&space)?);
    Ok(())
}
