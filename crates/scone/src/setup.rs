//! `scone setup <client>`: plug and play (gap-analysis P1). Zero
//! questions: detect the binary, write the config, say what happened.

use std::path::{Path, PathBuf};

/// Merge scone's MCP server entry into a Claude Desktop config, preserving
/// everything already there. Pure function, unit-tested.
pub fn merged_desktop_config(existing: &str, exe: &Path, space: &str) -> Result<String, String> {
    let mut root: serde_json::Value = if existing.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(existing).map_err(|e| format!("existing config is not JSON: {e}"))?
    };
    if !root.is_object() {
        return Err("existing config is not a JSON object".into());
    }
    let servers = root
        .as_object_mut()
        .expect("checked object")
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    if !servers.is_object() {
        return Err("mcpServers is not an object".into());
    }
    servers.as_object_mut().expect("checked object").insert(
        "scone".to_owned(),
        serde_json::json!({
            "command": exe.display().to_string(),
            "args": ["--space", space, "mcp"],
        }),
    );
    serde_json::to_string_pretty(&root).map_err(|e| e.to_string())
}

pub fn desktop_config_path() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("cannot resolve HOME")?;
    let base = if cfg!(target_os = "macos") {
        PathBuf::from(&home).join("Library/Application Support/Claude")
    } else {
        PathBuf::from(&home).join(".config/Claude")
    };
    Ok(base.join("claude_desktop_config.json"))
}

pub fn setup_claude_desktop(space: &str) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let path = desktop_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let merged = merged_desktop_config(&existing, &exe, space)?;
    std::fs::write(&path, merged).map_err(|e| e.to_string())?;
    Ok(format!(
        "wrote {}\nrestart Claude Desktop to pick up the scone memory server",
        path.display()
    ))
}

pub fn setup_claude_code(space: &str) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let output = std::process::Command::new("claude")
        .args([
            "mcp",
            "add",
            "scone",
            "--",
            &exe.display().to_string(),
            "--space",
            space,
            "mcp",
        ])
        .output()
        .map_err(|_| "the `claude` CLI is not on PATH; install Claude Code first".to_owned())?;
    if !output.status.success() {
        return Err(format!(
            "claude mcp add failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(format!(
        "registered the scone memory server with Claude Code (space: {space})"
    ))
}

/// Merge scone's hook wiring into a Claude Code settings.json, preserving
/// everything else. Pure function, unit-tested.
pub fn merged_settings_hooks(existing: &str, exe: &Path, space: &str) -> Result<String, String> {
    let mut root: serde_json::Value = if existing.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(existing).map_err(|e| format!("settings.json is not JSON: {e}"))?
    };
    if !root.is_object() {
        return Err("settings.json is not a JSON object".into());
    }
    let entry = |event: &str, timeout: u64| {
        serde_json::json!([{
            "hooks": [{
                "type": "command",
                "command": format!(
                    "{} --space {} hook {}",
                    exe.display(), space, event
                ),
                "timeout": timeout,
            }]
        }])
    };
    let hooks = root
        .as_object_mut()
        .expect("checked object")
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));
    if !hooks.is_object() {
        return Err("hooks is not an object".into());
    }
    let hooks = hooks.as_object_mut().expect("checked object");
    hooks.insert("SessionStart".into(), entry("session-start", 10));
    hooks.insert("UserPromptSubmit".into(), entry("user-prompt", 10));
    hooks.insert("SessionEnd".into(), entry("session-end", 60));
    serde_json::to_string_pretty(&root).map_err(|e| e.to_string())
}

/// Wire this project's `.claude/settings.json` to the scone hook handlers.
pub fn setup_claude_code_hooks(space: &str) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let dir = std::env::current_dir()
        .map_err(|e| e.to_string())?
        .join(".claude");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("settings.json");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let merged = merged_settings_hooks(&existing, &exe, space)?;
    std::fs::write(&path, merged).map_err(|e| e.to_string())?;
    Ok(format!(
        "wrote {}\nnew Claude Code sessions here get memory injection and capture (space: {space})",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_into_empty_and_invalid() {
        let merged =
            merged_desktop_config("", Path::new("/usr/local/bin/scone"), "default").unwrap();
        let v: serde_json::Value = serde_json::from_str(&merged).unwrap();
        assert_eq!(v["mcpServers"]["scone"]["args"][2], "mcp");
        assert!(merged_desktop_config("[1,2]", Path::new("/x"), "d").is_err());
        assert!(merged_desktop_config("not json", Path::new("/x"), "d").is_err());
    }
}
