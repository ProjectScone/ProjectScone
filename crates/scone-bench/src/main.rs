//! LongMemEval harness runner. `fetch` downloads the dataset; `run`
//! executes a subset against the engine with the configured LLM.

use clap::{Parser, Subcommand};
use scone_bench::{Report, parse_dataset};
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
    /// Generate a synthetic-from-real dataset: real prose carries the
    /// sessions, injected facts give exact ground truth
    Synth {
        /// File of real published text to harvest from
        #[arg(long)]
        source: std::path::PathBuf,
        #[arg(long, default_value_t = 40)]
        items: usize,
        #[arg(long, default_value_t = 6)]
        sessions: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Output path (LongMemEval-format JSON)
        #[arg(long, default_value = "bench-data/synthetic.json")]
        out: std::path::PathBuf,
    },
    /// Download the LongMemEval oracle split (smallest) to bench-data/
    Fetch {
        /// Override the dataset URL
        #[arg(
            long,
            default_value = "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle"
        )]
        url: String,
    },
    /// Grade answers produced by an outside reader against the dataset
    Score {
        /// The dataset the questions came from
        #[arg(long, default_value = "bench-data/longmemeval_s.json")]
        dataset: String,
        /// JSONL of {question_id, answer} from whatever did the reading
        answers: String,
        /// Judge endpoint (OpenAI-compatible); omit for substring only
        #[arg(long)]
        llm_url: Option<String>,
        /// Judge model at that endpoint
        #[arg(long)]
        llm_model: Option<String>,
    },
    /// Run the harness over a dataset file
    Run {
        /// Path to a LongMemEval-format JSON file
        #[arg(long, default_value = "bench-data/longmemeval_oracle.json")]
        dataset: String,
        /// Number of items (0 = all)
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Seed for the default stratified subset sampling
        #[arg(long, default_value_t = 42)]
        stratified_seed: u64,
        /// Take the head of the dataset instead of a stratified sample
        /// (the dataset is ordered by type; head runs measure one class)
        #[arg(long)]
        head: bool,
        /// Embedder: hash (hermetic) or local (ONNX, real semantics)
        #[arg(long, default_value = "local")]
        embedder: String,
        /// Local embed model: bge-small-en-v1.5 | bge-base-en-v1.5 |
        /// nomic-embed-text-v1.5
        #[arg(long, default_value = "bge-small-en-v1.5")]
        embed_model: String,
        /// OpenAI-compatible endpoint for the LLM (e.g. Ollama:
        /// http://localhost:11434/v1). Unset = episodic-only run.
        #[arg(long)]
        llm_url: Option<String>,
        /// Model name at the endpoint (required with --llm-url)
        #[arg(long)]
        llm_model: Option<String>,
        /// Also score with an LLM judge (same endpoint/model unless
        /// --judge-model overrides)
        #[arg(long)]
        judge: bool,
        /// Judge model override; pins the judge across reader experiments
        #[arg(long)]
        judge_model: Option<String>,
        /// Send think:false to the reader (Ollama reasoning models
        /// otherwise burn the budget on reasoning tokens); judge
        /// unaffected
        #[arg(long)]
        no_think: bool,
        /// Skip fact distillation: answer from raw episodic recall only
        #[arg(long)]
        no_distill: bool,
        /// Chunk target in bytes for ingestion (default: engine default)
        #[arg(long)]
        chunk_bytes: Option<usize>,
        /// Ingestion granularity: session (default) or turn
        #[arg(long, default_value = "session")]
        granularity: String,
        /// Attach the local cross-encoder reranker (bge-reranker-base)
        #[arg(long)]
        reranker: bool,
        /// Answer system prompt: v1 (default), v2 (extraction-style),
        /// or v3 (evidence-chaining; benchmarked 10 points WORSE than v2
        /// on an 8B reader)
        #[arg(long, default_value = "v1")]
        prompt: String,
        /// Retrieve for each clause of a multi-part question and fuse
        #[arg(long)]
        decompose: bool,
        /// Compute date arithmetic instead of generating it
        #[arg(long)]
        compute_temporal: bool,
        /// How many retrieved items to hand the reader (default 15)
        #[arg(long, default_value_t = 15)]
        recall_limit: usize,
        /// Write question and retrieved context per item to this JSONL
        /// file and skip answering. This is the subscription-native
        /// path: whatever agent you already pay for reads the pack and
        /// answers, then `score` grades it.
        #[arg(long)]
        handoff: Option<String>,
        /// Two-pass reader: pass 1 extracts evidence, pass 2 answers
        /// from only that evidence (benchmarked 13 points WORSE than
        /// single-pass on an 8B reader; temporal questions collapse)
        #[arg(long)]
        two_pass: bool,
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
        Cmd::Synth {
            source,
            items,
            sessions,
            seed,
            out,
        } => {
            let text = std::fs::read_to_string(&source).map_err(|e| e.to_string())?;
            let cfg = scone_bench::synth::SynthConfig {
                items,
                sessions_per_item: sessions,
                seed,
            };
            let json = scone_bench::synth::generate(&text, &cfg)?;
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&out, &json).map_err(|e| e.to_string())?;
            println!("wrote {} items to {}", items, out.display());
            Ok(())
        }
        Cmd::Score {
            dataset,
            answers,
            llm_url,
            llm_model,
        } => {
            let raw = std::fs::read_to_string(&dataset).map_err(|e| format!("{dataset}: {e}"))?;
            let items = parse_dataset(&raw)?;
            let by_id: std::collections::HashMap<&str, &scone_bench::BenchItem> =
                items.iter().map(|i| (i.question_id.as_str(), i)).collect();
            let judge = match (&llm_url, &llm_model) {
                (Some(url), Some(model)) => {
                    Some(scone_core::llm::OpenAiCompatible::new(url, model, None))
                }
                (None, None) => None,
                _ => return Err("--llm-url and --llm-model go together".into()),
            };
            let given = std::fs::read_to_string(&answers).map_err(|e| format!("{answers}: {e}"))?;
            let mut breakdown = scone_bench::TypeBreakdown::default();
            let (mut n, mut substring, mut judged) = (0usize, 0usize, 0usize);
            for line in given.lines().filter(|l| !l.trim().is_empty()) {
                let row: serde_json::Value =
                    serde_json::from_str(line).map_err(|e| format!("answers: {e}"))?;
                let id = row["question_id"].as_str().unwrap_or_default();
                let answer = row["answer"].as_str().unwrap_or_default();
                let Some(item) = by_id.get(id) else {
                    eprintln!("no such question in the dataset, skipped: {id}");
                    continue;
                };
                n += 1;
                let sub = scone_bench::substring_correct(&item.answer, answer);
                substring += usize::from(sub);
                let verdict = match &judge {
                    Some(llm) => scone_bench::judge_correct_typed(
                        llm,
                        &item.question_type,
                        &item.question,
                        &item.answer,
                        answer,
                    )?,
                    None => sub,
                };
                judged += usize::from(verdict);
                breakdown.add(&item.question_type, false, sub, Some(verdict));
                println!(
                    "[{n}] {id} {} {}",
                    if sub { "substring:YES" } else { "substring:NO" },
                    if verdict { "judge:YES" } else { "judge:NO" }
                );
            }
            if n == 0 {
                return Err("no answers matched the dataset".into());
            }
            println!("scored {n} answers");
            println!(
                "accuracy(substring): {:.1}%",
                substring as f64 / n as f64 * 100.0
            );
            println!("accuracy(judge): {:.1}%", judged as f64 / n as f64 * 100.0);
            print!("{}", breakdown.report());
            Ok(())
        }
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
            let name = url
                .rsplit('/')
                .next()
                .unwrap_or("dataset")
                .trim_end_matches(".json");
            let path = format!("bench-data/{name}.json");
            std::fs::write(&path, &body).map_err(|e| e.to_string())?;
            eprintln!("wrote {path}");
            eprintln!("saved {} MB", body.len() / (1 << 20));
            Ok(())
        }
        Cmd::Run {
            dataset,
            limit,
            stratified_seed,
            head,
            embedder,
            embed_model,
            llm_url,
            llm_model,
            judge,
            judge_model,
            no_think,
            no_distill,
            chunk_bytes,
            granularity,
            reranker,
            prompt,
            decompose,
            compute_temporal,
            recall_limit,
            handoff,
            two_pass,
        } => {
            let raw = std::fs::read_to_string(&dataset)
                .map_err(|e| format!("{dataset}: {e} (run `scone-bench fetch` first)"))?;
            let mut items = parse_dataset(&raw)?;
            if limit > 0 {
                if head {
                    items.truncate(limit);
                    eprintln!("head sample of {} (single-class risk)", items.len());
                } else {
                    items = scone_bench::stratified_sample(&items, limit, stratified_seed);
                    eprintln!(
                        "stratified sample of {} (seed {stratified_seed})",
                        items.len()
                    );
                }
            }
            let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
            let embedder: Box<dyn scone_core::embed::EmbeddingProvider> = match embedder.as_str() {
                "hash" => Box::new(HashEmbedder::new(384)),
                #[cfg(feature = "local-embed")]
                "local" => {
                    // Persistent cache: without it every run re-downloads the
                    // model, and one hung download stalls a whole sweep
                    // (observed 2026-08-27).
                    let cache = std::env::temp_dir().join("scone-bench-models");
                    Box::new(
                        scone_core::embed::OnnxEmbedder::with_model(&cache, &embed_model)
                            .map_err(|e| e.to_string())?,
                    )
                }
                other => return Err(format!("unknown embedder {other:?}")),
            };
            let mut engine = Engine::open(dir.path(), embedder).map_err(|e| e.to_string())?;
            if let Some(bytes) = chunk_bytes {
                engine.set_chunk_target(bytes);
            }
            #[cfg(feature = "local-embed")]
            if reranker {
                let cache = std::env::temp_dir().join("scone-bench-models");
                engine.set_reranker(Some(Box::new(
                    scone_core::rerank::OnnxReranker::new(&cache).map_err(|e| e.to_string())?,
                )));
                eprintln!("reranker: bge-reranker-base");
            }
            #[cfg(not(feature = "local-embed"))]
            if reranker {
                return Err("this build lacks local-embed; no reranker".into());
            }
            let granularity = match granularity.as_str() {
                "session" => scone_bench::Granularity::Session,
                "turn" => scone_bench::Granularity::Turn,
                other => return Err(format!("unknown granularity {other:?}")),
            };
            let answer_system = match prompt.as_str() {
                "v1" => None,
                "v2" => Some(scone_core::llm::ANSWER_SYSTEM_V2.to_owned()),
                "v3" => Some(scone_core::llm::ANSWER_SYSTEM_V3.to_owned()),
                other => return Err(format!("unknown prompt {other:?}")),
            };
            let run_opts = scone_bench::RunOpts {
                distill: !no_distill,
                granularity,
                answer_system,
                two_pass,
                decompose,
                compute_temporal,
                recall_limit,
            };
            match (&llm_url, &llm_model) {
                (Some(url), Some(model)) => {
                    let mut llm = scone_core::llm::OpenAiCompatible::new(url, model, None);
                    if no_think {
                        llm = llm.with_think(false);
                        eprintln!("llm: {model} at {url} (think off)");
                    } else {
                        eprintln!("llm: {model} at {url}");
                    }
                    engine.set_llm(Some(Box::new(llm)));
                }
                (None, None) => eprintln!("no LLM configured: episodic-only run"),
                _ => return Err("--llm-url and --llm-model go together".into()),
            }
            let judge_llm = if judge {
                match (&llm_url, judge_model.as_ref().or(llm_model.as_ref())) {
                    (Some(url), Some(model)) => {
                        eprintln!("judge: {model}");
                        Some(scone_core::llm::OpenAiCompatible::new(url, model, None))
                    }
                    _ => return Err("--judge needs --llm-url and --llm-model".into()),
                }
            } else {
                None
            };
            let mut report = Report::default();
            let mut breakdown = scone_bench::TypeBreakdown::default();
            let mut judged_correct = 0usize;
            let mut recall_ms = Vec::new();
            let mut r_any = [0usize; 3];
            let mut r_all = [0usize; 3];
            let mut handoff_out = String::new();
            for (i, item) in items.iter().enumerate() {
                let outcome = scone_bench::run_item_with(&mut engine, item, i, &run_opts)?;
                if handoff.is_some() {
                    // The reader must not see the answer it is being
                    // graded against, so only the question and what
                    // retrieval actually found travel across.
                    handoff_out.push_str(&format!(
                        "{}\n",
                        serde_json::json!({
                            "question_id": item.question_id,
                            "question_type": item.question_type,
                            "question_date": item.question_date,
                            "question": item.question,
                            "context": outcome.retrieved,
                        })
                    ));
                }
                recall_ms.push(outcome.recall_ms);
                for (slot, k) in [5usize, 10, 15].iter().enumerate() {
                    r_any[slot] += usize::from(outcome.recall_any_at(*k));
                    r_all[slot] += usize::from(outcome.recall_all_at(*k));
                }
                report.add(
                    outcome.correct,
                    outcome.stored_bytes,
                    outcome.retrieved_bytes,
                );
                let mut judged_verdict: Option<bool> = None;
                let judged = match &judge_llm {
                    Some(llm) => {
                        let ok = scone_bench::judge_correct_typed(
                            llm,
                            &item.question_type,
                            &item.question,
                            &item.answer,
                            &outcome.model_answer,
                        )?;
                        judged_correct += usize::from(ok);
                        judged_verdict = Some(ok);
                        if ok { " judge:YES" } else { " judge:NO" }
                    }
                    None => "",
                };
                breakdown.add(
                    &item.question_type,
                    outcome.recall_all_at(15),
                    outcome.correct,
                    judged_verdict,
                );
                let mut top_sessions: Vec<&str> = Vec::new();
                for s in &outcome.retrieved_sessions {
                    if !top_sessions.contains(&s.as_str()) {
                        top_sessions.push(s);
                    }
                }
                eprintln!(
                    "[{}/{}] {} {}{} ({:.1} ms recall) R@15:{} expected={:?} got_top={:?}",
                    i + 1,
                    items.len(),
                    outcome.question_id,
                    if outcome.correct { "correct" } else { "MISS" },
                    judged,
                    outcome.recall_ms,
                    if outcome.recall_all_at(15) {
                        "HIT"
                    } else {
                        "MISS"
                    },
                    outcome.answer_sessions,
                    top_sessions.iter().take(4).collect::<Vec<_>>(),
                );
            }
            recall_ms.sort_by(f64::total_cmp);
            let p50 = recall_ms.get(recall_ms.len() / 2).copied().unwrap_or(0.0);
            if judge_llm.is_some() {
                println!(
                    "accuracy(llm-judge): {:.1}%",
                    judged_correct as f64 / report.total.max(1) as f64 * 100.0
                );
            }
            print!("{}", breakdown.report());
            let pct = |n: usize| n as f64 / report.total.max(1) as f64 * 100.0;
            println!(
                "recall@5/10/15 any-evidence: {:.1}% / {:.1}% / {:.1}%   all-evidence: {:.1}% / {:.1}% / {:.1}%",
                pct(r_any[0]),
                pct(r_any[1]),
                pct(r_any[2]),
                pct(r_all[0]),
                pct(r_all[1]),
                pct(r_all[2]),
            );
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
            if let Some(path) = &handoff {
                std::fs::write(path, &handoff_out).map_err(|e| format!("{path}: {e}"))?;
                println!("handoff: wrote {} items to {path}", items.len());
                println!(
                    "answer each into a JSONL of {{question_id, answer}}, then: scone-bench score"
                );
            }
            Ok(())
        }
    }
}
