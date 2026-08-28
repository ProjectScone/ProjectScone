#![allow(clippy::unwrap_used)]
use assert_cmd::Command;

fn scone(dir: &std::path::Path) -> Command {
    let mut c = Command::cargo_bin("scone").unwrap();
    // Tests stay hermetic: the hash embedder needs no model download.
    c.arg("--data-dir").arg(dir).args(["--embedder", "hash"]);
    c
}

#[test]
fn doctor_rebuild_recovers_from_deleted_indexes() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "memory survives rebuilds"])
        .assert()
        .success();
    std::fs::remove_dir_all(dir.path().join("fts")).unwrap();
    scone(dir.path())
        .args(["doctor", "--rebuild"])
        .assert()
        .success()
        .stdout(predicates::str::contains("rebuilt"));
    scone(dir.path())
        .args(["search", "survives"])
        .assert()
        .success()
        .stdout(predicates::str::contains("survives"));
}

#[test]
fn add_then_search_finds_the_note() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the borrow checker enforces ownership"])
        .assert()
        .success()
        .stdout(predicates::str::contains("ingested"));
    scone(dir.path())
        .args(["search", "borrow checker"])
        .assert()
        .success()
        .stdout(predicates::str::contains("borrow checker"));
}

#[test]
fn duplicate_add_reports_deduplicated() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "same note"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "same note"])
        .assert()
        .success()
        .stdout(predicates::str::contains("deduplicated"));
}

#[test]
fn status_counts_episodes() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "one note"])
        .assert()
        .success();
    scone(dir.path())
        .arg("status")
        .assert()
        .success()
        .stdout(predicates::str::contains("episodes: 1"));
}

#[test]
fn spaces_are_isolated() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "secret in default"])
        .assert()
        .success();
    scone(dir.path())
        .args(["--space", "other", "search", "secret"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no results"));
}

#[test]
fn bad_space_name_is_a_clean_error() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["--space", "BAD NAME!", "add", "--note", "x"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("space name"));
}

#[test]
fn add_reads_files() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("doc.md");
    std::fs::write(&f, "# notes\n\nscone is a memory engine").unwrap();
    scone(dir.path()).arg("add").arg(&f).assert().success();
    scone(dir.path())
        .args(["search", "memory engine"])
        .assert()
        .success()
        .stdout(predicates::str::contains("doc.md"));
}

const FAKE_FACTS: &str = r#"[{"subject":"mark","predicate":"prefers","object":"bun"}]"#;
const FAKE_FACTS_2: &str = r#"[{"subject":"mark","predicate":"prefers","object":"pnpm"}]"#;

#[test]
fn status_says_semantic_lane_is_paused_without_llm() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "a note"])
        .assert()
        .success();
    scone(dir.path())
        .arg("status")
        .assert()
        .success()
        .stdout(predicates::str::contains("paused"))
        .stdout(predicates::str::contains("1 pending"));
}

#[test]
fn distill_extracts_facts_and_lists_them() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark said he prefers bun"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success()
        .stdout(predicates::str::contains("1 episode"));
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers bun"));
}

#[test]
fn contradiction_history_is_visible_and_explained() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "first claim"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS_2)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "second claim"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    let out = scone(dir.path()).args(["facts", "list"]).assert().success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(
        stdout.contains("bun") && !stdout.contains("pnpm"),
        "{stdout}"
    );
    scone(dir.path())
        .args(["facts", "list", "--all"])
        .assert()
        .success()
        .stdout(predicates::str::contains("pnpm"))
        .stdout(predicates::str::contains("superseded"));
}

#[test]
fn facts_why_shows_provenance_and_close_takes_reason() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark prefers bun today"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "why", "1"])
        .assert()
        .success()
        .stdout(predicates::str::contains("episode"));
    scone(dir.path())
        .args(["facts", "close", "1", "--reason", "no longer true"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "list", "--all"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no longer true"));
}

#[test]
fn ask_without_llm_prints_context_and_pause_notice() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy key lives in the vault"])
        .assert()
        .success();
    scone(dir.path())
        .args(["ask", "where is the deploy key?"])
        .assert()
        .success()
        .stdout(predicates::str::contains("deploy key"))
        .stdout(predicates::str::contains("paused"));
}

#[test]
fn ask_with_configured_llm_answers_from_the_stub() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 65536];
        let mut req = String::new();
        loop {
            let n = sock.read(&mut buf).unwrap();
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
            if req.contains("\r\n\r\n") && req.trim_end().ends_with('}') {
                break;
            }
        }
        let body = r#"{"choices":[{"message":{"content":"in the vault, per your note"}}]}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).unwrap();
    });
    std::fs::create_dir_all(dir.path()).unwrap();
    std::fs::write(
        dir.path().join("config.toml"),
        format!("[llm]\nprovider = \"openai\"\nbase_url = \"http://{addr}\"\nmodel = \"stub\"\n"),
    )
    .unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy key lives in the vault"])
        .assert()
        .success();
    scone(dir.path())
        .args(["ask", "where is the deploy key?"])
        .assert()
        .success()
        .stdout(predicates::str::contains("in the vault, per your note"));
    handle.join().unwrap();
}

#[test]
fn search_shows_facts_and_supports_as_of_time_travel() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "first claim about tools"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS_2)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "second claim about tools"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    // Pin intervals to known dates so as-of is deterministic.
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    db.execute(
        "UPDATE facts SET valid_from='2026-01-01T00:00:00Z', valid_until='2026-06-01T00:00:00Z'
         WHERE object='pnpm'",
        [],
    )
    .unwrap();
    db.execute(
        "UPDATE facts SET valid_from='2026-06-01T00:00:00Z' WHERE object='bun'",
        [],
    )
    .unwrap();
    drop(db);
    scone(dir.path())
        .args(["search", "mark prefers"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers bun"));
    scone(dir.path())
        .args(["search", "mark prefers", "--as-of", "2026-03-15T00:00:00Z"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers pnpm"));
}

#[test]
fn export_import_moves_memory_between_stores() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    scone(a.path())
        .args(["add", "--note", "portable memory survives moves"])
        .assert()
        .success();
    let out = scone(a.path()).arg("export").assert().success();
    let jsonl = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(jsonl.contains("portable memory"));
    let f = a.path().join("dump.jsonl");
    std::fs::write(&f, &jsonl).unwrap();
    scone(b.path())
        .arg("import")
        .arg(&f)
        .assert()
        .success()
        .stdout(predicates::str::contains("1 episode"));
    scone(b.path())
        .args(["search", "portable memory"])
        .assert()
        .success()
        .stdout(predicates::str::contains("survives moves"));
}

#[test]
fn spaces_lists_all_spaces() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "one"])
        .assert()
        .success();
    scone(dir.path())
        .args(["--space", "work", "add", "--note", "two"])
        .assert()
        .success();
    scone(dir.path())
        .arg("spaces")
        .assert()
        .success()
        .stdout(predicates::str::contains("default"))
        .stdout(predicates::str::contains("work"));
}

#[test]
fn watch_once_ingests_a_directory() {
    let dir = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::write(notes.path().join("idea.md"), "watch mode found this idea").unwrap();
    scone(dir.path())
        .arg("watch")
        .arg(notes.path())
        .arg("--once")
        .assert()
        .success()
        .stdout(predicates::str::contains("ingested 1"));
    scone(dir.path())
        .args(["search", "idea"])
        .assert()
        .success()
        .stdout(predicates::str::contains("found this idea"));
}

#[test]
fn daemon_once_scans_and_distills() {
    let dir = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::write(notes.path().join("fact.md"), "mark uses scone daily").unwrap();
    scone(dir.path())
        .env(
            "SCONE_FAKE_FACTS",
            r#"[{"subject":"mark","predicate":"uses","object":"scone"}]"#,
        )
        .args(["--llm", "fake", "daemon", "--once", "--watch"])
        .arg(notes.path())
        .assert()
        .success()
        .stdout(predicates::str::contains("distilled"));
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark uses scone"));
}

#[test]
fn profile_shows_identity_and_recent_activity() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "team switched to bun last sprint"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .arg("profile")
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers bun"))
        .stdout(predicates::str::contains("last sprint"));
}

#[test]
fn search_reports_context_economy() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..5 {
        scone(dir.path())
            .args([
                "add",
                "--note",
                &format!("filler note number {i} about various topics"),
            ])
            .assert()
            .success();
    }
    scone(dir.path())
        .args(["add", "--note", "the tokens saved line matters"])
        .assert()
        .success();
    scone(dir.path())
        .args(["search", "tokens saved", "--limit", "2"])
        .assert()
        .success()
        .stdout(predicates::str::contains("% saved"));
}

#[test]
fn tagging_focuses_search_and_curates_sources() {
    let dir = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::write(
        notes.path().join("paper.md"),
        "the retrieval paper describes fusion",
    )
    .unwrap();
    scone(dir.path())
        .args([
            "add",
            "--note",
            "meeting notes about fusion budget",
            "--tag",
            "meeting",
        ])
        .assert()
        .success();
    scone(dir.path())
        .arg("watch")
        .arg(notes.path())
        .args(["--once", "--tag", "research"])
        .assert()
        .success();
    // Auto extension tag curates the source kind.
    scone(dir.path())
        .arg("tags")
        .assert()
        .success()
        .stdout(predicates::str::contains("meeting"))
        .stdout(predicates::str::contains("research"))
        .stdout(predicates::str::contains("md"));
    scone(dir.path())
        .args(["search", "fusion", "--tag", "research"])
        .assert()
        .success()
        .stdout(predicates::str::contains("retrieval paper"));
    let out = scone(dir.path())
        .args(["search", "fusion", "--tag", "meeting"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(
        stdout.contains("budget") && !stdout.contains("retrieval paper"),
        "{stdout}"
    );
}

#[test]
fn setup_claude_desktop_writes_mcp_config_preserving_existing() {
    let dir = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let cfg_dir = if cfg!(target_os = "macos") {
        home.path().join("Library/Application Support/Claude")
    } else {
        home.path().join(".config/Claude")
    };
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("claude_desktop_config.json"),
        r#"{"mcpServers": {"other": {"command": "keepme"}}, "theme": "dark"}"#,
    )
    .unwrap();
    scone(dir.path())
        .env("HOME", home.path())
        .args(["setup", "claude-desktop"])
        .assert()
        .success()
        .stdout(predicates::str::contains("claude_desktop_config.json"));
    let written: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(cfg_dir.join("claude_desktop_config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        written["mcpServers"]["other"]["command"], "keepme",
        "existing servers kept"
    );
    assert_eq!(written["theme"], "dark", "unrelated settings kept");
    let scone_cfg = &written["mcpServers"]["scone"];
    assert!(scone_cfg["command"].as_str().unwrap().contains("scone"));
    assert!(
        scone_cfg["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "mcp")
    );
}

#[test]
fn setup_claude_code_invokes_the_claude_cli() {
    let dir = tempfile::tempdir().unwrap();
    let bin = tempfile::tempdir().unwrap();
    let log = bin.path().join("invocations.log");
    let shim = bin.path().join("claude");
    std::fs::write(
        &shim,
        format!("#!/bin/sh\necho \"$@\" >> {}\n", log.display()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = format!(
        "{}:{}",
        bin.path().display(),
        std::env::var("PATH").unwrap()
    );
    scone(dir.path())
        .env("PATH", &path)
        .args(["setup", "claude-code"])
        .assert()
        .success()
        .stdout(predicates::str::contains("registered"));
    let logged = std::fs::read_to_string(&log).unwrap();
    assert!(logged.contains("mcp add scone"), "{logged}");
    assert!(logged.contains("mcp"), "{logged}");
}

#[test]
fn hook_session_start_injects_profile_context() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "this project uses tokio and axum"])
        .assert()
        .success();
    scone(dir.path())
        .env(
            "SCONE_FAKE_FACTS",
            r#"[{"subject":"project","predicate":"uses","object":"axum"}]"#,
        )
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["hook", "session-start"])
        .write_stdin(r#"{"session_id": "s1", "cwd": "/tmp"}"#)
        .assert()
        .success()
        .stdout(predicates::str::contains("project uses axum"));
}

#[test]
fn hook_user_prompt_injects_relevant_memory_and_fails_open() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy password rotates every monday"])
        .assert()
        .success();
    scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin(r#"{"prompt": "when does the deploy password rotate?"}"#)
        .assert()
        .success()
        .stdout(predicates::str::contains("rotates every monday"));
    // Garbage stdin must not break the session: exit 0, empty stdout.
    let out = scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin("not json at all")
        .assert()
        .success();
    assert!(out.get_output().stdout.is_empty());
}

#[test]
fn hook_session_end_captures_the_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let transcript = dir.path().join("transcript.jsonl");
    std::fs::write(&transcript, concat!(
        r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"we decided to use blake3 for hashing"}]}}"#, "\n",
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"noted, blake3 it is"}]}}"#, "\n",
        r#"{"type":"other","ignored":true}"#, "\n",
    )).unwrap();
    scone(dir.path())
        .args(["hook", "session-end"])
        .write_stdin(format!(
            r#"{{"transcript_path": "{}"}}"#,
            transcript.display()
        ))
        .assert()
        .success();
    scone(dir.path())
        .args(["search", "blake3 hashing"])
        .assert()
        .success()
        .stdout(predicates::str::contains("blake3"));
    scone(dir.path())
        .arg("tags")
        .assert()
        .success()
        .stdout(predicates::str::contains("claude-code"));
}

#[test]
fn setup_claude_code_hooks_wires_project_settings() {
    let dir = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project.path().join(".claude")).unwrap();
    std::fs::write(
        project.path().join(".claude/settings.json"),
        r#"{"permissions": {"allow": ["Bash(ls:*)"]}}"#,
    )
    .unwrap();
    scone(dir.path())
        .current_dir(project.path())
        .args(["setup", "claude-code-hooks"])
        .assert()
        .success()
        .stdout(predicates::str::contains("settings.json"));
    let written: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(project.path().join(".claude/settings.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        written["permissions"]["allow"][0], "Bash(ls:*)",
        "existing settings kept"
    );
    for event in ["SessionStart", "UserPromptSubmit", "SessionEnd"] {
        let cmd = written["hooks"][event][0]["hooks"][0]["command"]
            .as_str()
            .unwrap_or_default();
        assert!(cmd.contains("hook"), "{event}: {cmd}");
        assert!(cmd.contains("scone"), "{event}: {cmd}");
    }
}
