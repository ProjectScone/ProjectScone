//! `scone hook <event>`: Claude Code hook handlers, all local, all
//! fail-open. The predecessor's plugin runs Node scripts with 3-second
//! network caps; ours is the engine binary itself on millisecond budgets
//! (design: docs/superpowers/specs/2026-08-28-team-brain-concept.md).
//!
//! Contract: stdout becomes injected context (SessionStart /
//! UserPromptSubmit); any internal failure exits 0 with empty stdout so a
//! broken store never stalls a session. Errors go to stderr.

use std::path::{Path, PathBuf};

use scone_core::{Engine, RecallOpts, auth};

pub fn session_start(engine: &mut Engine, space_name: &str) -> String {
    let mut run = || -> Result<String, String> {
        let space = auth::resolve(engine, space_name, true).map_err(|e| e.to_string())?;
        let profile = engine.profile(&space, 6).map_err(|e| e.to_string())?;
        let mut out = String::new();
        if !profile.static_facts.is_empty() {
            out.push_str("Persistent memory for this project:\n");
            for f in &profile.static_facts {
                out.push_str(&format!("- {} {} {}\n", f.subject, f.predicate, f.object));
            }
        }
        if !profile.dynamic.is_empty() {
            out.push_str("Recent activity:\n");
            for d in profile.dynamic.iter().take(3) {
                out.push_str(&format!("- {}\n", d.replace('\n', " ")));
            }
        }
        Ok(out)
    };
    run().unwrap_or_else(|e| {
        eprintln!("scone hook session-start: {e}");
        String::new()
    })
}

/// Cosine similarity below which a memory is not worth a prompt's
/// tokens, per embedder. Measured with bge-small-en-v1.5: genuine
/// matches land at 0.68 to 0.73, while queries about nothing in the
/// store top out at 0.49 to 0.55, so the gap is real and 0.60 sits
/// inside it. Recall itself keeps returning everything; only
/// unattended injection is filtered, because a person who searches
/// wants their weak hits.
///
/// An unmeasured embedder gets no floor. A number calibrated against
/// one model says nothing about another, and silently withholding
/// memory is a worse failure than spending tokens on a weak match.
fn min_similarity(embedder_id: &str) -> Option<f32> {
    match embedder_id {
        "bge-small-en-v1.5" => Some(0.60),
        _ => None,
    }
}

/// How many recalled memories to remember per session, so the same
/// ones are not re-injected on every prompt.
const SESSION_MEMORY: usize = 500;

/// Where a session's already-injected memories are remembered.
fn seen_path(data_dir: &Path, session_id: &str) -> Option<PathBuf> {
    // Session ids come from the host agent; never let one escape the
    // directory by carrying separators or dots.
    let safe: String = session_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect();
    (!safe.is_empty()).then(|| data_dir.join("hook-sessions").join(format!("{safe}.txt")))
}

fn load_seen(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn save_seen(path: &Path, seen: &[String]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tail = seen.len().saturating_sub(SESSION_MEMORY);
    let _ = std::fs::write(path, seen[tail..].join("\n"));
}

/// What a piece of recalled text is, ignoring whitespace noise.
fn fingerprint(text: &str) -> String {
    let normalized: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    blake3::hash(normalized.to_lowercase().as_bytes()).to_hex()[..16].to_owned()
}

pub fn user_prompt(engine: &mut Engine, space_name: &str, data_dir: &Path, stdin: &str) -> String {
    let mut run = || -> Result<String, String> {
        let value: serde_json::Value = serde_json::from_str(stdin).map_err(|e| e.to_string())?;
        let prompt = value["prompt"].as_str().ok_or("no prompt field")?;
        if prompt.trim().len() < 8 {
            return Ok(String::new());
        }
        let space = auth::resolve(engine, space_name, true).map_err(|e| e.to_string())?;
        let pack = engine
            .recall(
                &space,
                prompt,
                &RecallOpts {
                    limit: 3,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
        if pack.facts.is_empty() && pack.items.is_empty() {
            return Ok(String::new());
        }
        let floor = min_similarity(engine.embedder_id());
        let session = value["session_id"].as_str().unwrap_or_default();
        let path = seen_path(data_dir, session);
        let mut seen = path.as_deref().map(load_seen).unwrap_or_default();

        let mut lines = Vec::new();
        let mut repeats = 0usize;
        let mut weak = 0usize;
        for f in &pack.facts {
            // Facts are the distilled answer, not a passage: they are
            // small, and dropping them for a rank would lose the point.
            let line = format!("{} {} {}", f.subject, f.predicate, f.object);
            let print = fingerprint(&line);
            if seen.contains(&print) {
                repeats += 1;
                continue;
            }
            seen.push(print);
            lines.push(format!("- {line}"));
        }
        for item in &pack.items {
            if let (Some(floor), Some(similarity)) = (floor, item.similarity)
                && similarity < floor
            {
                weak += 1;
                continue;
            }
            let text: String = item.text.chars().take(300).collect();
            let print = fingerprint(&text);
            if seen.contains(&print) {
                repeats += 1;
                continue;
            }
            seen.push(print);
            lines.push(format!("- [{}] {}", item.day(), text.replace('\n', " ")));
        }
        if lines.is_empty() {
            // Nothing new and nothing close enough: say nothing rather
            // than spend the prompt on a header.
            if let Some(path) = &path {
                save_seen(path, &seen);
            }
            return Ok(String::new());
        }
        if let Some(path) = &path {
            save_seen(path, &seen);
        }
        let mut out = String::from("Relevant memory:\n");
        out.push_str(&lines.join("\n"));
        out.push('\n');
        if repeats > 0 || weak > 0 {
            out.push_str(&format!(
                "({} new, {repeats} already in context, {weak} below the relevance floor)\n",
                lines.len()
            ));
        }
        Ok(out)
    };
    run().unwrap_or_else(|e| {
        eprintln!("scone hook user-prompt: {e}");
        String::new()
    })
}

pub fn session_end(engine: &mut Engine, space_name: &str, stdin: &str) {
    let mut run = || -> Result<(), String> {
        let value: serde_json::Value = serde_json::from_str(stdin).map_err(|e| e.to_string())?;
        let path = value["transcript_path"]
            .as_str()
            .ok_or("no transcript_path")?;
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut lines = Vec::new();
        for line in raw.lines() {
            let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let role = entry["message"]["role"].as_str().unwrap_or_default();
            if role != "user" && role != "assistant" {
                continue;
            }
            let Some(parts) = entry["message"]["content"].as_array() else {
                continue;
            };
            for part in parts {
                if let Some(text) = part["text"].as_str()
                    && !text.trim().is_empty()
                {
                    lines.push(format!("{role}: {text}"));
                }
            }
        }
        if lines.is_empty() {
            return Ok(());
        }
        let space = auth::resolve(engine, space_name, true).map_err(|e| e.to_string())?;
        let (episode_id, _fresh) = engine
            .import_episode(&space, "conversation", &lines.join("\n"), None, None)
            .map_err(|e| e.to_string())?;
        engine
            .tag_episode(&space, episode_id, &["claude-code"])
            .map_err(|e| e.to_string())?;
        if engine.has_llm() {
            let _ = engine.distill(&space, 25);
        }
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("scone hook session-end: {e}");
    }
}
