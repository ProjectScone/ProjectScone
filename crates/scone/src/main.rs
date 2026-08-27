//! `scone` — the CLI face of the Scone memory engine.
//!
//! Thin by design: every capability lives in `scone-core`; this binary
//! parses arguments, prints, and sets the exit code.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use scone_core::embed::{EmbeddingProvider, HashEmbedder};
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
    },
    /// Hybrid search within the space
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Show stores, counts, and index health
    Status,
    /// Verify and repair the derived indexes
    Doctor {
        /// Rebuild all indexes from SQLite truth
        #[arg(long)]
        rebuild: bool,
    },
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

    match &cli.cmd {
        Cmd::Add { paths, note } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let mut inputs = Vec::new();
            if let Some(text) = note {
                inputs.push(IngestInput::Note { text: text.clone() });
            }
            for path in paths {
                inputs.push(IngestInput::File { path: path.clone() });
            }
            if inputs.is_empty() {
                return Err("nothing to add: pass file paths or --note".into());
            }
            for input in inputs {
                let label = match &input {
                    IngestInput::Note { .. } => "note".to_owned(),
                    IngestInput::File { path } => path.display().to_string(),
                };
                match engine.ingest(&space, input).map_err(|e| e.to_string())? {
                    IngestOutcome::Ingested { episode_id, chunks } => {
                        println!("ingested {label} as episode {episode_id} ({chunks} chunks)");
                    }
                    IngestOutcome::Deduplicated { episode_id } => {
                        println!("deduplicated {label}: already stored as episode {episode_id}");
                    }
                }
            }
        }
        Cmd::Search { query, limit } => {
            let space = auth::resolve(&mut engine, &cli.space, true).map_err(|e| e.to_string())?;
            let opts = RecallOpts {
                limit: *limit,
                budget_bytes: None,
                ..Default::default()
            };
            let pack = engine
                .recall(&space, query, &opts)
                .map_err(|e| e.to_string())?;
            for warning in &pack.degraded {
                eprintln!("degraded: {warning}");
            }
            if pack.items.is_empty() {
                println!("no results");
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
        Cmd::Status => {
            let report = engine.status().map_err(|e| e.to_string())?;
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
