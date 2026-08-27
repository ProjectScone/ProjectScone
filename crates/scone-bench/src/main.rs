//! LongMemEval harness runner. `fetch` downloads the dataset; `run`
//! executes a subset against the engine with the configured LLM.

use clap::{Parser, Subcommand};
use scone_bench::{Report, parse_dataset, run_item};
use scone_core::Engine;
use scone_core::embed::HashEmbedder;

#[derive(Parser)]
#[command(
    name = "scone-bench",
    about = "LongMemEval harness for the Scone engine"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the LongMemEval oracle split (smallest) to bench-data/
    Fetch {
        /// Override the dataset URL
        #[arg(
            long,
            default_value = "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle.json"
        )]
        url: String,
    },
    /// Run the harness over a dataset file
    Run {
        /// Path to a LongMemEval-format JSON file
        #[arg(long, default_value = "bench-data/longmemeval_oracle.json")]
        dataset: String,
        /// Number of items (0 = all)
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Embedder: hash (hermetic) or local (ONNX, real semantics)
        #[arg(long, default_value = "local")]
        embedder: String,
        /// OpenAI-compatible endpoint for the LLM (e.g. Ollama:
        /// http://localhost:11434/v1). Unset = episodic-only run.
        #[arg(long)]
        llm_url: Option<String>,
        /// Model name at the endpoint (required with --llm-url)
        #[arg(long)]
        llm_model: Option<String>,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::Fetch { url } => {
            std::fs::create_dir_all("bench-data").map_err(|e| e.to_string())?;
            eprintln!("fetching {url} …");
            let mut res = ureq::get(&url).call().map_err(|e| e.to_string())?;
            let body = res
                .body_mut()
                .with_config()
                .limit(1 << 30)
                .read_to_vec()
                .map_err(|e| e.to_string())?;
            std::fs::write("bench-data/longmemeval_oracle.json", &body)
                .map_err(|e| e.to_string())?;
            eprintln!("saved {} MB", body.len() / (1 << 20));
            Ok(())
        }
        Cmd::Run {
            dataset,
            limit,
            embedder,
            llm_url,
            llm_model,
        } => {
            let raw = std::fs::read_to_string(&dataset)
                .map_err(|e| format!("{dataset}: {e} (run `scone-bench fetch` first)"))?;
            let mut items = parse_dataset(&raw)?;
            if limit > 0 {
                items.truncate(limit);
            }
            let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
            let embedder: Box<dyn scone_core::embed::EmbeddingProvider> = match embedder.as_str() {
                "hash" => Box::new(HashEmbedder::new(384)),
                #[cfg(feature = "local-embed")]
                "local" => Box::new(
                    scone_core::embed::OnnxEmbedder::new(&dir.path().join("models"))
                        .map_err(|e| e.to_string())?,
                ),
                other => return Err(format!("unknown embedder {other:?}")),
            };
            let mut engine = Engine::open(dir.path(), embedder).map_err(|e| e.to_string())?;
            match (&llm_url, &llm_model) {
                (Some(url), Some(model)) => {
                    engine.set_llm(Some(Box::new(scone_core::llm::OpenAiCompatible::new(
                        url, model, None,
                    ))));
                    eprintln!("llm: {model} at {url}");
                }
                (None, None) => eprintln!("no LLM configured: episodic-only run"),
                _ => return Err("--llm-url and --llm-model go together".into()),
            }
            let mut report = Report::default();
            let mut recall_ms = Vec::new();
            for (i, item) in items.iter().enumerate() {
                let outcome = run_item(&mut engine, item, i)?;
                recall_ms.push(outcome.recall_ms);
                report.add(
                    outcome.correct,
                    outcome.stored_bytes,
                    outcome.retrieved_bytes,
                );
                eprintln!(
                    "[{}/{}] {} {} ({:.1} ms recall)",
                    i + 1,
                    items.len(),
                    outcome.question_id,
                    if outcome.correct { "correct" } else { "MISS" },
                    outcome.recall_ms,
                );
            }
            recall_ms.sort_by(f64::total_cmp);
            let p50 = recall_ms.get(recall_ms.len() / 2).copied().unwrap_or(0.0);
            println!(
                "items: {}  accuracy(substring): {:.1}%  context-reduction: {:.1}%  recall p50: {:.1} ms",
                report.total,
                report.accuracy() * 100.0,
                report.context_reduction() * 100.0,
                p50,
            );
            println!(
                "note: substring scoring under-counts paraphrases; LLM-judge scoring is a follow-up"
            );
            Ok(())
        }
    }
}
