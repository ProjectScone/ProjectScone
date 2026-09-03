//! `scone hook <event>`: Claude Code hook handlers, all local, all
//! fail-open. The predecessor's plugin runs Node scripts with 3-second
//! network caps; ours is the engine binary itself on millisecond budgets
//! (design: docs/superpowers/specs/2026-08-28-team-brain-concept.md).
//!
//! Contract: stdout becomes injected context (SessionStart /
//! UserPromptSubmit); any internal failure exits 0 with empty stdout so a
//! broken store never stalls a session. Errors go to stderr.

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

pub fn user_prompt(engine: &mut Engine, space_name: &str, stdin: &str) -> String {
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
        let mut out = String::from("Relevant memory:\n");
        for f in &pack.facts {
            out.push_str(&format!("- {} {} {}\n", f.subject, f.predicate, f.object));
        }
        for item in &pack.items {
            let text: String = item.text.chars().take(300).collect();
            out.push_str(&format!("- [{}] {}\n", item.day(), text.replace('\n', " ")));
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
