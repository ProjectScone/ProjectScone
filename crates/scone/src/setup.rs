//! `scone setup <client>`: plug and play (gap-analysis P1). Zero
//! questions: detect the binary, write the config, say what happened.

use std::path::{Path, PathBuf};

/// How a client spells "here is a stdio MCP server". The clients agree on
/// the idea and disagree on every detail, so the shape is data.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// `{"mcpServers": {"scone": {"command", "args"}}}`
    /// Claude Desktop, Cursor, Windsurf, Gemini CLI.
    McpServers,
    /// `{"servers": {"scone": {"type": "stdio", "command", "args"}}}`
    /// VS Code's mcp.json.
    VsCodeServers,
    /// `{"context_servers": {"scone": {"source": "custom", "enabled", ...}}}`
    /// Zed's settings.json.
    ZedContextServers,
    /// `[mcp_servers.scone]` with `command` and `args`. Codex CLI's TOML.
    CodexToml,
}

/// The server entry itself, in whichever dialect the client reads.
fn entry(shape: Shape, exe: &Path, space: &str) -> serde_json::Value {
    let command = exe.display().to_string();
    let args = serde_json::json!(["--space", space, "mcp"]);
    match shape {
        Shape::McpServers => serde_json::json!({"command": command, "args": args}),
        Shape::VsCodeServers => {
            serde_json::json!({"type": "stdio", "command": command, "args": args})
        }
        Shape::ZedContextServers => serde_json::json!({
            "source": "custom", "enabled": true, "command": command, "args": args
        }),
        Shape::CodexToml => serde_json::json!({"command": command, "args": args}),
    }
}

/// The top-level key a client keeps its servers under.
fn container(shape: Shape) -> &'static str {
    match shape {
        Shape::McpServers => "mcpServers",
        Shape::VsCodeServers => "servers",
        Shape::ZedContextServers => "context_servers",
        Shape::CodexToml => "mcp_servers",
    }
}

/// Merge scone into any JSON-shaped client config, preserving everything
/// already there (other servers, unrelated settings, user edits).
/// Pure function, unit-tested against every shape.
pub fn merged_json_config(
    existing: &str,
    shape: Shape,
    exe: &Path,
    space: &str,
) -> Result<String, String> {
    let mut root: serde_json::Value = if existing.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(existing).map_err(|e| format!("existing config is not JSON: {e}"))?
    };
    if !root.is_object() {
        return Err("existing config is not a JSON object".into());
    }
    let key = container(shape);
    let servers = root
        .as_object_mut()
        .expect("checked object")
        .entry(key)
        .or_insert_with(|| serde_json::json!({}));
    if !servers.is_object() {
        return Err(format!("{key} is not an object"));
    }
    servers
        .as_object_mut()
        .expect("checked object")
        .insert("scone".to_owned(), entry(shape, exe, space));
    serde_json::to_string_pretty(&root).map_err(|e| e.to_string())
}

/// Merge scone into Codex CLI's TOML config, preserving the rest of the
/// file's tables. Pure function, unit-tested.
pub fn merged_toml_config(existing: &str, exe: &Path, space: &str) -> Result<String, String> {
    let mut root: toml::Table = if existing.trim().is_empty() {
        toml::Table::new()
    } else {
        existing
            .parse()
            .map_err(|e| format!("existing config is not TOML: {e}"))?
    };
    let servers = root
        .entry("mcp_servers")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let servers = servers
        .as_table_mut()
        .ok_or("mcp_servers is not a table in the existing config")?;
    let mut scone = toml::Table::new();
    scone.insert(
        "command".into(),
        toml::Value::String(exe.display().to_string()),
    );
    scone.insert(
        "args".into(),
        toml::Value::Array(vec![
            toml::Value::String("--space".into()),
            toml::Value::String(space.to_owned()),
            toml::Value::String("mcp".into()),
        ]),
    );
    servers.insert("scone".into(), toml::Value::Table(scone));
    toml::to_string_pretty(&root).map_err(|e| e.to_string())
}

/// Merge scone's MCP server entry into a Claude Desktop config, preserving
/// everything already there. Pure function, unit-tested.
pub fn merged_desktop_config(existing: &str, exe: &Path, space: &str) -> Result<String, String> {
    merged_json_config(existing, Shape::McpServers, exe, space)
}

/// A client scone can register itself with by editing a config file.
pub struct Client {
    /// What the user types after `scone setup`.
    pub name: &'static str,
    /// Config path relative to the user's home directory.
    pub rel_path: &'static str,
    pub shape: Shape,
    /// Shown after a successful write.
    pub after: &'static str,
}

/// Every client whose config layout we have verified. Adding one is a
/// row here, not a new code path.
pub const CLIENTS: &[Client] = &[
    Client {
        name: "cursor",
        rel_path: ".cursor/mcp.json",
        shape: Shape::McpServers,
        after: "restart Cursor, then check Settings > MCP",
    },
    Client {
        name: "windsurf",
        rel_path: ".codeium/windsurf/mcp_config.json",
        shape: Shape::McpServers,
        after: "restart Windsurf, then refresh MCP servers in Cascade",
    },
    Client {
        name: "gemini-cli",
        rel_path: ".gemini/settings.json",
        shape: Shape::McpServers,
        after: "restart the gemini CLI",
    },
    Client {
        name: "zed",
        rel_path: ".config/zed/settings.json",
        shape: Shape::ZedContextServers,
        after: "restart Zed; scone appears under context servers",
    },
    Client {
        name: "codex",
        rel_path: ".codex/config.toml",
        shape: Shape::CodexToml,
        after: "restart the codex CLI",
    },
    Client {
        name: "vscode",
        rel_path: ".vscode/mcp.json",
        shape: Shape::VsCodeServers,
        after: "reload VS Code; start the server from the mcp.json gutter",
    },
];

pub fn client_by_name(name: &str) -> Option<&'static Client> {
    CLIENTS.iter().find(|c| c.name == name)
}

/// Register scone with any known client: read what is there, merge, write
/// back. Never clobbers a config it cannot parse.
pub fn setup_client(client: &Client, space: &str) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let home = std::env::var_os("HOME").ok_or("cannot resolve HOME")?;
    let path = PathBuf::from(&home).join(client.rel_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let merged = match client.shape {
        Shape::CodexToml => merged_toml_config(&existing, &exe, space)?,
        shape => merged_json_config(&existing, shape, &exe, space)?,
    };
    std::fs::write(&path, merged).map_err(|e| e.to_string())?;
    Ok(format!(
        "wrote {}\n{} (space: {space})",
        path.display(),
        client.after
    ))
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

    /// Every client's dialect, from an empty config and from one that
    /// already holds another server. A setup command that eats a user's
    /// existing MCP servers is worse than no setup command.
    #[test]
    fn every_shape_merges_without_losing_what_was_there() {
        let exe = Path::new("/usr/local/bin/scone");
        for client in CLIENTS {
            if client.shape == Shape::CodexToml {
                let existing = "[mcp_servers.other]\ncommand = \"other\"\nargs = []\n";
                let merged = merged_toml_config(existing, exe, "work").unwrap();
                let v: toml::Table = merged.parse().unwrap();
                let servers = v["mcp_servers"].as_table().unwrap();
                assert!(
                    servers.contains_key("other"),
                    "{}: dropped other",
                    client.name
                );
                assert_eq!(servers["scone"]["args"][1].as_str(), Some("work"));
                continue;
            }
            let key = container(client.shape);
            let existing = format!("{{\"{key}\": {{\"other\": {{\"command\": \"x\"}}}}}}");
            let merged = merged_json_config(&existing, client.shape, exe, "work").unwrap();
            let v: serde_json::Value = serde_json::from_str(&merged).unwrap();
            assert!(
                v[key]["other"].is_object(),
                "{}: dropped the existing server",
                client.name
            );
            assert_eq!(
                v[key]["scone"]["command"].as_str(),
                Some("/usr/local/bin/scone"),
                "{}: no scone entry",
                client.name
            );
            assert_eq!(v[key]["scone"]["args"][1].as_str(), Some("work"));
        }
    }

    #[test]
    fn vscode_and_zed_carry_their_required_fields() {
        let exe = Path::new("/usr/local/bin/scone");
        let vs = merged_json_config("", Shape::VsCodeServers, exe, "d").unwrap();
        let v: serde_json::Value = serde_json::from_str(&vs).unwrap();
        assert_eq!(v["servers"]["scone"]["type"].as_str(), Some("stdio"));
        let zed = merged_json_config("", Shape::ZedContextServers, exe, "d").unwrap();
        let z: serde_json::Value = serde_json::from_str(&zed).unwrap();
        assert_eq!(
            z["context_servers"]["scone"]["source"].as_str(),
            Some("custom")
        );
        assert_eq!(z["context_servers"]["scone"]["enabled"], true);
    }

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
