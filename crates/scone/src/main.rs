//! `scone` — the CLI face of the Scone memory engine.
//!
//! Thin by design: every capability lives in `scone-core`; this binary
//! parses arguments, prints, and sets the exit code.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use scone_core::embed::{EmbeddingProvider, HashEmbedder};
use scone_core::llm::{AnthropicProvider, ExtractedFact, FakeLlm, LlmProvider, OpenAiCompatible};
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

const HASH_EMBEDDER_DIM: usize = 256;

#[derive(Parser)]
#[command(
    name = "scone",
    about = "A local-first temporal memory engine",
    version
)]
struct Cli {
    /// Data directory (default: ~/.scone)
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    /// Space to operate in
    #[arg(long, global = true, default_value = "default")]
    space: String,
    /// Embedding provider: local ONNX model (default) or the model-free
    /// deterministic hash embedder
    #[arg(long, global = true, value_enum, default_value_t = EmbedderKind::Local)]
    embedder: EmbedderKind,
    /// LLM for the semantic lane: none (paused) or fake (testing hook
    /// reading SCONE_FAKE_FACTS as a JSON fact array)
    #[arg(long, global = true, value_enum, default_value_t = LlmKind::Config)]
    llm: LlmKind,
    /// Attach the local cross-encoder reranker for higher recall precision
    /// (downloads bge-reranker-base once into the data dir)
    #[arg(long, global = true)]
    reranker: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Ingest files, or a note via --note
    Add {
        /// Files to ingest
        paths: Vec<PathBuf>,
        /// Ingest this text directly as a note
        #[arg(long)]
        note: Option<String>,
        /// Fetch and ingest a web page (converted to markdown locally)
        #[arg(long)]
        url: Option<String>,
        /// Tag the ingested content (repeatable); files also get their
        /// extension as a source tag
        #[arg(long = "tag")]
        tags: Vec<String>,
    },
    /// Hybrid search within the space
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Evaluate fact validity at this ISO-8601 instant (time travel)
        #[arg(long)]
        as_of: Option<String>,
        /// Focus on episodes carrying ALL of these tags (repeatable)
        #[arg(long = "tag")]
        tags: Vec<String>,
    },
    /// Claude Code hook handler (reads hook JSON on stdin, fail-open)
    Hook {
        /// session-start, user-prompt, or session-end
        event: String,
    },
    /// Register scone as a memory server, zero questions asked
    Setup {
        /// claude-code, claude-desktop, claude-code-hooks, cursor,
        /// windsurf, vscode, codex, zed, or gemini-cli
        client: String,
    },
    /// List tags in this space with usage counts
    Tags,
    /// Show the space's profile: identity facts and recent activity
    Profile {
        #[arg(long, default_value_t = 8)]
        limit: usize,
    },
    /// Show stores, counts, and index health
    Status,
    /// Scan a directory into memory, repeatedly or once
    Watch {
        /// Directory to scan
        dir: std::path::PathBuf,
        /// Scan once and exit (default: rescan every --interval-secs)
        #[arg(long)]
        once: bool,
        #[arg(long, default_value_t = 60)]
        interval_secs: u64,
        /// Tag everything ingested by this watch (repeatable)
        #[arg(long = "tag")]
        tags: Vec<String>,
    },
    /// Background loop: scan watch dirs and distill pending episodes
    Daemon {
        /// Directories to scan each cycle
        #[arg(long)]
        watch: Vec<std::path::PathBuf>,
        /// Run one cycle and exit
        #[arg(long)]
        once: bool,
        #[arg(long, default_value_t = 300)]
        interval_secs: u64,
    },
    /// Export the space as JSONL to stdout (episodes, aliases, facts)
    Export,
    /// Import a JSONL export into the space (idempotent)
    Import {
        /// Path to a JSONL file, or - for stdin
        path: std::path::PathBuf,
    },
    /// List spaces with their episode counts
    Spaces,
    /// Store a connector credential (notion). Token from --token or the
    /// SCONE_<PROVIDER>_TOKEN environment variable.
    Connect {
        /// Connector name, e.g. notion
        provider: String,
        /// Access token; omit to use the environment variable
        #[arg(long)]
        token: Option<String>,
    },
    /// Pull new documents from connected services into memory
    Sync {
        /// Connector to sync; omit to sync everything connected
        provider: Option<String>,
        /// Re-read everything, ignoring the last sync cursor
        #[arg(long)]
        full: bool,
    },
    /// Serve persistent agent memory over MCP (stdio)
    Mcp,
    /// Open a local console in your browser to read and correct memory
    Ui {
        /// Port on loopback (default 7438)
        #[arg(long, default_value_t = 7438)]
        port: u16,
        /// Print the URL instead of opening a browser
        #[arg(long)]
        no_open: bool,
    },
    /// Serve the multi-user HTTP API (keys from config.toml [[server.keys]])
    Serve {
        /// Listen address (overrides config [server].listen)
        #[arg(long)]
        listen: Option<String>,
    },
    /// Verify and repair the derived indexes
    Doctor {
        /// Rebuild all indexes from SQLite truth
        #[arg(long)]
        rebuild: bool,
    },
    /// Ask a question against your memory (recall + optional LLM answer)
    Ask {
        question: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Distill pending episodes into temporal facts (needs --llm)
    Distill {
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Inspect the semantic store
    Facts {
        #[command(subcommand)]
        action: FactsCmd,
    },
}

#[derive(Subcommand)]
enum FactsCmd {
    /// List facts (active only; --all includes closed with reasons)
    List {
        #[arg(long)]
        all: bool,
    },
    /// Show which episodes taught us a fact
    Why { id: i64 },
    /// Close a fact by hand with a reason (never deletes)
    Close {
        id: i64,
        #[arg(long)]
        reason: String,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum LlmKind {
    /// Read [llm] from <data-dir>/config.toml (absent file = none)
    Config,
    None,
    Fake,
}

fn llm_from_config(dir: &std::path::Path) -> Result<Option<Box<dyn LlmProvider>>, String> {
    let path = dir.join("config.toml");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Ok(None);
    };
    let table: toml::Table = raw
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let Some(llm) = table.get("llm") else {
        return Ok(None);
    };
    let get = |key: &str| llm.get(key).and_then(|v| v.as_str()).map(str::to_owned);
    let provider = get("provider").unwrap_or_else(|| "none".into());
    let api_key = match get("api_key_env") {
        Some(env_name) => Some(std::env::var(&env_name).map_err(|_| {
            format!("config names api_key_env = {env_name}, but that variable is not set")
        })?),
        None => None,
    };
    let model = get("model");
    let need_model = || {
        model
            .clone()
            .ok_or_else(|| "config [llm] needs model".to_owned())
    };
    match provider.as_str() {
        "none" => Ok(None),
        "ollama" => Ok(Some(Box::new(OpenAiCompatible::new(
            &get("base_url").unwrap_or_else(|| "http://localhost:11434/v1".into()),
            &need_model()?,
            api_key,
        )))),
        "openai" => Ok(Some(Box::new(OpenAiCompatible::new(
            &get("base_url").unwrap_or_else(|| "https://api.openai.com/v1".into()),
            &need_model()?,
            api_key,
        )))),
        "anthropic" => Ok(Some(Box::new(AnthropicProvider::new(
            &get("base_url").unwrap_or_else(|| "https://api.anthropic.com".into()),
            &need_model()?,
            &api_key.ok_or("anthropic needs api_key_env in config")?,
        )))),
        other => Err(format!("unknown llm provider {other:?} in config")),
    }
}

fn make_llm(kind: LlmKind, dir: &std::path::Path) -> Result<Option<Box<dyn LlmProvider>>, String> {
    match kind {
        LlmKind::Config => llm_from_config(dir),
        LlmKind::None => Ok(None),
        LlmKind::Fake => {
            let raw = std::env::var("SCONE_FAKE_FACTS").unwrap_or_else(|_| "[]".into());
            let parsed: serde_json::Value =
                serde_json::from_str(&raw).map_err(|e| format!("SCONE_FAKE_FACTS: {e}"))?;
            let facts = parsed
                .as_array()
                .ok_or("SCONE_FAKE_FACTS must be a JSON array")?
                .iter()
                .map(|f| {
                    Ok(ExtractedFact {
                        subject: f["subject"]
                            .as_str()
                            .ok_or("fact needs subject")?
                            .to_owned(),
                        predicate: f["predicate"]
                            .as_str()
                            .ok_or("fact needs predicate")?
                            .to_owned(),
                        object: f["object"].as_str().ok_or("fact needs object")?.to_owned(),
                        confidence: f["confidence"].as_f64().unwrap_or(0.8) as f32,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Some(Box::new(FakeLlm::new(facts))))
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum EmbedderKind {
    Local,
    Hash,
}

fn make_embedder(
    kind: EmbedderKind,
    dir: &std::path::Path,
) -> Result<Box<dyn EmbeddingProvider>, String> {
    match kind {
        EmbedderKind::Hash => Ok(Box::new(HashEmbedder::new(HASH_EMBEDDER_DIM))),
        #[cfg(feature = "local-embed")]
        EmbedderKind::Local => Ok(Box::new(
            scone_core::embed::OnnxEmbedder::new(&dir.join("models")).map_err(|e| e.to_string())?,
        )),
        #[cfg(not(feature = "local-embed"))]
        EmbedderKind::Local => {
            let _ = dir;
            Err("this build lacks the local-embed feature; use --embedder hash".into())
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Enough entropy for a per-run console key without pulling in a
/// random-number crate: the clock and this process's own address space.
fn session_key_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = u64::from(std::process::id());
    let stack = &nanos as *const u64 as u64;
    nanos.rotate_left(17).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ pid.wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stack.rotate_left(31)
}

fn data_dir(cli: &Cli) -> Result<PathBuf, String> {
    if let Some(d) = &cli.data_dir {
        return Ok(d.clone());
    }
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".scone"))
        .ok_or_else(|| "cannot resolve home directory; pass --data-dir".to_owned())
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let dir = data_dir(&cli)?;
    let embedder = make_embedder(cli.embedder, &dir)?;
    let repair = matches!(cli.cmd, Cmd::Doctor { rebuild: true });
    let mut engine = if repair {
        Engine::open_for_repair(&dir, embedder).map_err(|e| e.to_string())?
    } else {
        Engine::open(&dir, embedder).map_err(|e| e.to_string())?
    };
    engine.set_llm(make_llm(cli.llm, &dir)?);
    #[cfg(feature = "local-embed")]
    if cli.reranker {
        engine.set_reranker(Some(Box::new(
            scone_core::rerank::OnnxReranker::new(&dir.join("models"))
                .map_err(|e| e.to_string())?,
        )));
    }

    match &cli.cmd {
        Cmd::Add {
            paths,
            note,
            url,
            tags,
        } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let mut inputs = Vec::new();
            if let Some(text) = note {
                inputs.push(IngestInput::Note { text: text.clone() });
            }
            for path in paths {
                inputs.push(IngestInput::File { path: path.clone() });
            }
            if let Some(url) = url {
                let (markdown, domain) = scone::web::fetch_page(url)?;
                let (episode_id, fresh) = engine
                    .import_episode(&space, "file", &markdown, Some(url), None)
                    .map_err(|e| e.to_string())?;
                println!(
                    "{} {url} as episode {episode_id}",
                    if fresh { "ingested" } else { "deduplicated" }
                );
                let mut all: Vec<&str> = tags.iter().map(String::as_str).collect();
                all.push("url");
                all.push(&domain);
                engine
                    .tag_episode(&space, episode_id, &all)
                    .map_err(|e| e.to_string())?;
            }
            if inputs.is_empty() && url.is_none() {
                return Err("nothing to add: pass file paths, --note, or --url".into());
            }
            for input in inputs {
                let label = match &input {
                    IngestInput::Note { .. } => "note".to_owned(),
                    IngestInput::File { path } => path.display().to_string(),
                };
                let is_file = matches!(&input, IngestInput::File { .. });
                let outcome = engine.ingest(&space, input).map_err(|e| e.to_string())?;
                let episode_id = match outcome {
                    IngestOutcome::Ingested { episode_id, chunks } => {
                        println!("ingested {label} as episode {episode_id} ({chunks} chunks)");
                        episode_id
                    }
                    IngestOutcome::Deduplicated { episode_id } => {
                        println!("deduplicated {label}: already stored as episode {episode_id}");
                        episode_id
                    }
                };
                let mut all: Vec<String> = tags.clone();
                if is_file
                    && let Some(ext) = std::path::Path::new(&label)
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                {
                    all.push(ext);
                }
                if !all.is_empty() {
                    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
                    engine
                        .tag_episode(&space, episode_id, &refs)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Cmd::Search {
            query,
            limit,
            as_of,
            tags,
        } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let opts = RecallOpts {
                limit: *limit,
                budget_bytes: None,
                as_of: as_of.clone(),
                tags: tags.clone(),
                ..Default::default()
            };
            let pack = engine
                .recall(&space, query, &opts)
                .map_err(|e| e.to_string())?;
            for warning in &pack.degraded {
                eprintln!("degraded: {warning}");
            }
            for f in &pack.facts {
                println!(
                    "fact  {} {} {}  (conf {:.2}, {})",
                    f.subject, f.predicate, f.object, f.confidence, f.status
                );
            }
            if pack.items.is_empty() && pack.facts.is_empty() {
                println!("no results");
            } else {
                println!(
                    "context: {} bytes of {} stored ({:.1}% saved)",
                    pack.returned_bytes,
                    pack.space_bytes,
                    pack.context_reduction() * 100.0
                );
            }
            for item in &pack.items {
                let text: String = item.text.chars().take(120).collect();
                let text = text.replace('\n', " ");
                let source = item.source.as_deref().unwrap_or("note");
                println!(
                    "{:.3}  [{}] {}  {}",
                    item.score, item.episode_id, source, text
                );
            }
        }
        Cmd::Hook { event } => {
            use std::io::Read;
            let mut stdin = String::new();
            let _ = std::io::stdin().read_to_string(&mut stdin);
            match event.as_str() {
                "session-start" => {
                    print!("{}", scone::hook::session_start(&mut engine, &cli.space));
                }
                "user-prompt" => {
                    print!(
                        "{}",
                        scone::hook::user_prompt(&mut engine, &cli.space, &stdin)
                    );
                }
                "session-end" => {
                    scone::hook::session_end(&mut engine, &cli.space, &stdin);
                }
                other => return Err(format!("unknown hook event {other:?}")),
            }
        }
        Cmd::Setup { client } => {
            let message = match client.as_str() {
                "claude-code" => scone::setup::setup_claude_code(&cli.space)?,
                "claude-desktop" => scone::setup::setup_claude_desktop(&cli.space)?,
                "claude-code-hooks" => scone::setup::setup_claude_code_hooks(&cli.space)?,
                other => match scone::setup::client_by_name(other) {
                    Some(client) => scone::setup::setup_client(client, &cli.space)?,
                    None => {
                        let known: Vec<&str> = scone::setup::CLIENTS
                            .iter()
                            .map(|c| c.name)
                            .chain(["claude-code", "claude-desktop", "claude-code-hooks"])
                            .collect();
                        return Err(format!(
                            "unknown client {other:?}: try one of {}",
                            known.join(", ")
                        ));
                    }
                },
            };
            println!("{message}");
        }
        Cmd::Tags => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let tags = engine.tags_list(&space).map_err(|e| e.to_string())?;
            if tags.is_empty() {
                println!("no tags yet: add with `scone add --tag <name>`");
            }
            for (name, count) in tags {
                println!("{name}  ({count})");
            }
        }
        Cmd::Profile { limit } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let profile = engine.profile(&space, *limit).map_err(|e| e.to_string())?;
            if !profile.static_facts.is_empty() {
                println!("profile:");
                for f in &profile.static_facts {
                    println!(
                        "  {} {} {}  (conf {:.2})",
                        f.subject, f.predicate, f.object, f.confidence
                    );
                }
            }
            if !profile.dynamic.is_empty() {
                println!("recent:");
                for d in &profile.dynamic {
                    println!("  {}", d.replace('\n', " "));
                }
            }
            if profile.static_facts.is_empty() && profile.dynamic.is_empty() {
                println!("empty profile: nothing stored in this space yet");
            }
        }
        Cmd::Status => {
            let report = engine.status().map_err(|e| e.to_string())?;
            if report.read_only {
                println!("READ-ONLY: another scone process holds the write lock");
            }
            for s in &report.spaces {
                println!(
                    "space {}: episodes: {} chunks: {} revision: {}",
                    s.name, s.episodes, s.chunks, s.revision
                );
            }
            println!(
                "embedder: {} ({} dims)",
                report.embedder_id, report.embedder_dim
            );
            println!(
                "indexes: {}",
                if report.index_dirty {
                    "DIRTY — run `scone doctor --rebuild`"
                } else {
                    "clean"
                }
            );
            match &report.llm_id {
                Some(id) => println!(
                    "semantic lane: llm {id} — {} pending, {} failed",
                    report.pending_distill, report.failed_distill
                ),
                None => println!(
                    "semantic lane: paused — no LLM configured ({} pending, {} failed)",
                    report.pending_distill, report.failed_distill
                ),
            }
        }
        Cmd::Ask { question, limit } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let opts = RecallOpts {
                limit: *limit,
                expand_neighbors: true,
                ..Default::default()
            };
            let pack = engine
                .recall(&space, question, &opts)
                .map_err(|e| e.to_string())?;
            let mut context = String::new();
            for f in &pack.facts {
                context.push_str(&format!(
                    "- fact: {} {} {}\n",
                    f.subject, f.predicate, f.object
                ));
            }
            for item in &pack.items {
                context.push_str(&format!(
                    "- [{} | episode {}] {}\n",
                    item.day(),
                    item.episode_id,
                    item.text
                ));
            }
            if engine.has_llm() {
                let answer = engine
                    .llm_answer(question, &context)
                    .map_err(|e| e.to_string())?;
                println!("{answer}");
                let ids: Vec<String> = pack
                    .items
                    .iter()
                    .map(|i| i.episode_id.to_string())
                    .collect();
                if !ids.is_empty() {
                    println!("\nsources: episodes {}", ids.join(", "));
                }
            } else {
                if context.is_empty() {
                    println!("no matching memory");
                } else {
                    print!("{context}");
                }
                println!("(semantic lane paused — no LLM configured; showing raw recall)");
            }
        }
        Cmd::Distill { limit } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let r = engine.distill(&space, *limit).map_err(|e| e.to_string())?;
            println!(
                "distilled {} episode{}: +{} facts, {} closed, {} failed",
                r.processed,
                if r.processed == 1 { "" } else { "s" },
                r.facts_added,
                r.facts_closed,
                r.failed
            );
        }
        Cmd::Facts { action } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            match action {
                FactsCmd::List { all } => {
                    let facts = engine.facts_list(&space, *all).map_err(|e| e.to_string())?;
                    if facts.is_empty() {
                        println!("no facts");
                    }
                    for f in facts {
                        println!(
                            "[{}] {} {} {}  (conf {:.2}, {}, from {})",
                            f.fact_id,
                            f.subject,
                            f.predicate,
                            f.object,
                            f.confidence,
                            f.status,
                            f.valid_from
                        );
                    }
                }
                FactsCmd::Why { id } => {
                    for p in engine.facts_why(&space, *id).map_err(|e| e.to_string())? {
                        println!(
                            "episode {} ({}) {} at {}",
                            p.episode_id,
                            p.kind,
                            p.source.as_deref().unwrap_or("note"),
                            p.created_at
                        );
                    }
                }
                FactsCmd::Close { id, reason } => {
                    engine
                        .facts_close(&space, *id, reason)
                        .map_err(|e| e.to_string())?;
                    println!("closed fact {id}: {reason}");
                }
            }
        }
        Cmd::Watch {
            dir: watch_dir,
            once,
            interval_secs,
            tags,
        } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            loop {
                let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                let r = engine
                    .ingest_directory_tagged(&space, watch_dir, 1_000_000, &tag_refs)
                    .map_err(|e| e.to_string())?;
                println!(
                    "ingested {} ({} unchanged, {} skipped) from {}",
                    r.ingested,
                    r.deduplicated,
                    r.skipped,
                    watch_dir.display()
                );
                if *once {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs((*interval_secs).max(1)));
            }
        }
        Cmd::Daemon {
            watch,
            once,
            interval_secs,
        } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            loop {
                for watch_dir in watch {
                    let r = engine
                        .ingest_directory(&space, watch_dir, 1_000_000)
                        .map_err(|e| e.to_string())?;
                    println!(
                        "scanned {}: {} new, {} unchanged, {} skipped",
                        watch_dir.display(),
                        r.ingested,
                        r.deduplicated,
                        r.skipped
                    );
                }
                if engine.has_llm() {
                    let r = engine.distill(&space, 100).map_err(|e| e.to_string())?;
                    println!(
                        "distilled {}: +{} facts, {} closed, {} failed",
                        r.processed, r.facts_added, r.facts_closed, r.failed
                    );
                } else {
                    println!("distilled 0: semantic lane paused (no LLM configured)");
                }
                let expired = engine.decay_facts(&space, 90).map_err(|e| e.to_string())?;
                if expired > 0 {
                    println!("decayed {expired} stale facts (reasons recorded)");
                }
                if *once {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs((*interval_secs).max(1)));
            }
        }
        Cmd::Export => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let jsonl = engine.export_jsonl(&space).map_err(|e| e.to_string())?;
            print!("{jsonl}");
        }
        Cmd::Connect { provider, token } => {
            if !scone::connectors::KNOWN.contains(&provider.as_str()) {
                return Err(format!(
                    "unknown connector {provider:?}: known connectors are {}",
                    scone::connectors::KNOWN.join(", ")
                ));
            }
            let mut creds = scone::connectors::load_credentials(&dir)?;
            let token = match token {
                Some(t) => t.clone(),
                None => scone::connectors::token_for(provider, &creds).ok_or_else(|| {
                    format!(
                        "no token: pass --token or set SCONE_{}_TOKEN",
                        provider.to_uppercase()
                    )
                })?,
            };
            let entry = creds.providers.entry(provider.clone()).or_default();
            entry.token = token;
            scone::connectors::save_credentials(&dir, &creds)?;
            println!(
                "connected {provider}; credential stored in {} (0600)",
                scone::connectors::credentials_path(&dir).display()
            );
            println!("run `scone sync {provider}` to pull documents into memory");
        }
        Cmd::Sync { provider, full } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let mut creds = scone::connectors::load_credentials(&dir)?;
            let wanted: Vec<String> = match provider {
                Some(p) => vec![p.clone()],
                None => scone::connectors::KNOWN
                    .iter()
                    .map(|p| (*p).to_owned())
                    .filter(|p| scone::connectors::token_for(p, &creds).is_some())
                    .collect(),
            };
            if wanted.is_empty() {
                println!("nothing connected: run `scone connect <provider>` first");
                return Ok(());
            }
            for name in &wanted {
                let Some(token) = scone::connectors::token_for(name, &creds) else {
                    println!("{name}: not connected, skipping");
                    continue;
                };
                let connector = scone::connectors::connector_for(name, token)?;
                let since = if *full {
                    None
                } else {
                    creds.providers.get(name).and_then(|p| p.last_sync.clone())
                };
                let docs = connector.fetch(since.as_deref())?;
                let (mut added, mut seen) = (0usize, 0usize);
                let mut newest = since.clone();
                for doc in &docs {
                    let content = if doc.title.is_empty() {
                        doc.body.clone()
                    } else {
                        format!("{}\n\n{}", doc.title, doc.body)
                    };
                    let source = if doc.url.is_empty() {
                        format!("{name}:{}", doc.id)
                    } else {
                        doc.url.clone()
                    };
                    let (episode_id, fresh) = engine
                        .import_episode(
                            &space,
                            "connector",
                            &content,
                            Some(&source),
                            doc.updated_at.as_deref(),
                        )
                        .map_err(|e| e.to_string())?;
                    engine
                        .tag_episode(&space, episode_id, &[name.as_str()])
                        .map_err(|e| e.to_string())?;
                    if fresh {
                        added += 1;
                    } else {
                        seen += 1;
                    }
                    let advances = matches!(
                        &doc.updated_at,
                        Some(edited) if newest.as_deref().is_none_or(|n| edited.as_str() > n)
                    );
                    if advances {
                        newest.clone_from(&doc.updated_at);
                    }
                }
                // Only advance the cursor on success, so an interrupted
                // sync re-reads rather than silently skipping documents.
                if let Some(newest) = newest {
                    creds.providers.entry(name.clone()).or_default().last_sync = Some(newest);
                }
                println!("{name}: {added} new, {seen} already known");
            }
            scone::connectors::save_credentials(&dir, &creds)?;
        }
        Cmd::Import { path } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let data = if path.as_os_str() == "-" {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin()
                    .read_to_string(&mut buf)
                    .map_err(|e| e.to_string())?;
                buf
            } else {
                std::fs::read_to_string(path).map_err(|e| e.to_string())?
            };
            let report = engine
                .import_jsonl(&space, &data)
                .map_err(|e| e.to_string())?;
            println!(
                "imported {} episode{} ({} deduplicated), {} fact{}, {} alias{}",
                report.episodes,
                if report.episodes == 1 { "" } else { "s" },
                report.deduplicated,
                report.facts,
                if report.facts == 1 { "" } else { "s" },
                report.aliases,
                if report.aliases == 1 { "" } else { "es" },
            );
        }
        Cmd::Spaces => {
            let report = engine.status().map_err(|e| e.to_string())?;
            if report.spaces.is_empty() {
                println!("no spaces yet");
            }
            for s in &report.spaces {
                println!(
                    "{}  ({} episodes, revision {})",
                    s.name, s.episodes, s.revision
                );
            }
        }
        Cmd::Mcp => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            let server = scone::mcp::SconeMcp::new(engine, &cli.space);
            rt.block_on(async move {
                use rmcp::ServiceExt;
                let running = server
                    .serve(rmcp::transport::stdio())
                    .await
                    .map_err(|e| e.to_string())?;
                running.waiting().await.map_err(|e| e.to_string())?;
                Ok::<(), String>(())
            })?;
            return Ok(());
        }
        Cmd::Ui { port, no_open } => {
            // A fresh key per run, never written down. The console is
            // loopback-only, so the only way to reach it is to already
            // be on this machine, but the auth chokepoint stays intact
            // rather than growing an unauthenticated path.
            let key = format!("ui-{:016x}", session_key_seed());
            let space = cli.space.clone();
            let addr = format!("127.0.0.1:{port}");
            let app = scone::serve::console_router(
                engine,
                scone::serve::ServeConfig {
                    keys: vec![scone::serve::SpaceKey {
                        key: key.clone(),
                        space: space.clone(),
                    }],
                },
                &key,
            );
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .map_err(|e| format!("bind {addr}: {e}"))?;
                let url = format!("http://{addr}/");
                println!("scone console for space {space:?} at {url}");
                println!("stop it with ctrl-c; the key dies with this process");
                if !*no_open {
                    // Best effort: a failed opener must not stop the server.
                    let opener = if cfg!(target_os = "macos") {
                        "open"
                    } else {
                        "xdg-open"
                    };
                    let _ = std::process::Command::new(opener).arg(&url).status();
                }
                axum::serve(listener, app).await.map_err(|e| e.to_string())
            })?;
            return Ok(());
        }
        Cmd::Serve { listen } => {
            let raw = std::fs::read_to_string(dir.join("config.toml"))
                .map_err(|_| "scone serve needs config.toml with [[server.keys]]".to_owned())?;
            let table: toml::Table = raw.parse().map_err(|e| format!("config.toml: {e}"))?;
            let server = table
                .get("server")
                .ok_or("config.toml needs a [server] section")?;
            let keys: Vec<scone::serve::SpaceKey> = server
                .get("keys")
                .and_then(|k| k.as_array())
                .map(|rows| {
                    rows.iter()
                        .filter_map(|row| {
                            Some(scone::serve::SpaceKey {
                                key: row.get("key")?.as_str()?.to_owned(),
                                space: row.get("space")?.as_str()?.to_owned(),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            if keys.is_empty() {
                return Err(
                    "refusing to serve with zero API keys — add [[server.keys]]                      entries (key, space) to config.toml"
                        .into(),
                );
            }
            let addr = listen
                .clone()
                .or_else(|| {
                    server
                        .get("listen")
                        .and_then(|l| l.as_str())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "127.0.0.1:7437".to_owned());
            let n_keys = keys.len();
            let app = scone::serve::router(engine, scone::serve::ServeConfig { keys });
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .map_err(|e| format!("bind {addr}: {e}"))?;
                println!("scone serving on http://{addr} ({n_keys} keys)");
                axum::serve(listener, app).await.map_err(|e| e.to_string())
            })?;
            return Ok(());
        }
        Cmd::Doctor { rebuild } => {
            if !rebuild {
                let report = engine.status().map_err(|e| e.to_string())?;
                println!(
                    "indexes: {}",
                    if report.index_dirty { "DIRTY" } else { "clean" }
                );
                println!("run `scone doctor --rebuild` to rebuild from truth");
            } else {
                let report = engine.doctor_rebuild().map_err(|e| e.to_string())?;
                println!(
                    "rebuilt: {} episodes, {} chunks ({} re-embedded)",
                    report.episodes, report.chunks, report.reembedded
                );
            }
        }
    }
    Ok(())
}
