#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A small code-retrieval probe over this repository's own Rust sources:
//! twenty questions phrased the way a developer asks, each with the
//! function that answers it. Measures Recall@5 and Recall@10 of a chunk
//! overlapping that function's span, with and without contextual code
//! embedding. Needs the ONNX embedder, so it is ignored by default:
//!
//!   cargo test -p scone-core --test code_probe -- --ignored --nocapture
//!
//! Results are recorded in memory/EXPERIMENTS.md, never quoted from
//! memory. The questions avoid the function's own name where a plain
//! reader would; the point is whether a body cut from its signature can
//! still be found.
//!
//! Two question sets. DOC_QUESTIONS were written by the author of the
//! doc comments, and E31 found they paraphrase them (median 70% word
//! overlap). COMMIT_QUESTIONS are the subject lines of the commits that
//! introduced the same functions, lifted verbatim from git log, so they
//! were phrased at a different time for a different purpose; a commit
//! that introduced several of the functions accepts any of them.
//! Select with SCONE_PROBE=commits (default: doc).
use scone_core::chunker::{Syntax, chunk_syntax};
use scone_core::embed::OnnxEmbedder;
use scone_core::{Engine, RecallOpts, auth};

/// One question and the functions that answer it (file basename, name).
struct Probe {
    question: &'static str,
    targets: &'static [(&'static str, &'static str)],
}

/// Written by the author of the doc comments; see E31.
const DOC_QUESTIONS: [Probe; 20] = [
    Probe {
        question: "close the older facts a newly active fact supersedes, or close it if something newer is on record",
        targets: &[("distill.rs", "settle_active_fact")],
    },
    Probe {
        question: "answer a temporal question by computing over dated episodes, or decline and leave it to the reader",
        targets: &[("temporal.rs", "answer_temporally")],
    },
    Probe {
        question: "does this line open a declaration, walking past modifiers like pub(crate)",
        targets: &[("chunker.rs", "starts_declaration")],
    },
    Probe {
        question: "pick prose or code cutting from a source path's extension",
        targets: &[("chunker.rs", "syntax_for")],
    },
    Probe {
        question: "register another name for an entity",
        targets: &[("distill.rs", "add_entity_alias")],
    },
    Probe {
        question: "which episodes taught us a fact",
        targets: &[("distill.rs", "facts_why")],
    },
    Probe {
        question: "close a fact by hand with a reason without deleting it",
        targets: &[("distill.rs", "facts_close")],
    },
    Probe {
        question: "list the proposed facts waiting for a person",
        targets: &[("distill.rs", "facts_pending")],
    },
    Probe {
        question: "turn a relative date phrase like a week ago into a window anchored at now",
        targets: &[("temporal.rs", "relative_window")],
    },
    Probe {
        question: "number of days between two dates",
        targets: &[("temporal.rs", "days_between")],
    },
    Probe {
        question: "resolve a space name to a scoped space, creating it when allowed",
        targets: &[("auth.rs", "resolve")],
    },
    Probe {
        question: "recall memory for an agent through the MCP tool with the profile prepended",
        targets: &[("mcp.rs", "memory_recall")],
    },
    Probe {
        question: "the HTTP handler for recall with query parameters",
        targets: &[("serve.rs", "get_recall")],
    },
    Probe {
        question: "draw a stratified sample of benchmark items with a seed",
        targets: &[("lib.rs", "stratified_sample")],
    },
    Probe {
        question: "rebuild the indexes from the database when they are missing or corrupt",
        targets: &[("lib.rs", "doctor_rebuild")],
    },
    Probe {
        question: "attach tags to an episode",
        targets: &[("tags.rs", "tag_episode")],
    },
    Probe {
        question: "the profile of a space: identity facts and recent activity",
        targets: &[("profile.rs", "profile")],
    },
    Probe {
        question: "apply the facts extracted from one episode in one transaction, deduplicating restatements",
        targets: &[("distill.rs", "apply_facts")],
    },
    Probe {
        question: "read a question as a temporal operator or decide it is not one",
        targets: &[("temporal.rs", "plan")],
    },
    Probe {
        question: "split a multi-part question into clauses to retrieve for",
        targets: &[("recall.rs", "decompose")],
    },
];

/// Subject lines of the commits that introduced the target functions,
/// verbatim from git log (the first commit whose diff adds the fn).
const COMMIT_QUESTIONS: [Probe; 20] = [
    Probe {
        question: "Hold extracted facts below a confidence gate for review",
        targets: &[
            ("distill.rs", "settle_active_fact"),
            ("distill.rs", "facts_pending"),
        ],
    },
    Probe {
        question: "Compute the date arithmetic instead of generating it",
        targets: &[
            ("temporal.rs", "answer_temporally"),
            ("temporal.rs", "days_between"),
            ("temporal.rs", "plan"),
        ],
    },
    Probe {
        question: "Cut code at declarations instead of blank lines",
        targets: &[
            ("chunker.rs", "starts_declaration"),
            ("chunker.rs", "syntax_for"),
            ("chunker.rs", "chunk_syntax"),
        ],
    },
    Probe {
        question: "Apply facts with entities, provenance, and contradiction closure",
        targets: &[
            ("distill.rs", "add_entity_alias"),
            ("distill.rs", "apply_facts"),
        ],
    },
    Probe {
        question: "Add distill and facts commands with loud lane status",
        targets: &[("distill.rs", "facts_why"), ("distill.rs", "facts_close")],
    },
    Probe {
        question: "Resolve the dates people say instead of the ones they write",
        targets: &[("temporal.rs", "relative_window")],
    },
    Probe {
        question: "Add auth module minting ScopedSpace handles (invariant I5)",
        targets: &[("auth.rs", "resolve")],
    },
    Probe {
        question: "Serve agent memory over MCP with bounded, scoped tools",
        targets: &[("mcp.rs", "memory_recall")],
    },
    Probe {
        question: "Serve the multi-user HTTP API with per-key space scoping",
        targets: &[("serve.rs", "get_recall")],
    },
    Probe {
        question: "Sample subsets stratified by question type",
        targets: &[("lib.rs", "stratified_sample")],
    },
    Probe {
        question: "Add local ONNX embeddings and doctor rebuild",
        targets: &[("lib.rs", "doctor_rebuild")],
    },
    Probe {
        question: "Add user tags with focused recall (schema v3)",
        targets: &[("tags.rs", "tag_episode")],
    },
    Probe {
        question: "Add two-tier profiles: identity facts plus recent activity",
        targets: &[("profile.rs", "profile")],
    },
    Probe {
        question: "Retrieve each clause of a question, not just the whole",
        targets: &[("recall.rs", "decompose")],
    },
    Probe {
        question: "Embed the doc comment with a code chunk, and say what the probe can prove",
        targets: &[
            ("chunker.rs", "doc_comment_above"),
            ("chunker.rs", "enclosing_doc_comment"),
        ],
    },
    Probe {
        question: "Resolve \"last Friday\" and \"in February\"",
        targets: &[
            ("timeparse.rs", "days_in_month"),
            ("temporal.rs", "year_month"),
        ],
    },
    Probe {
        question: "Order only the events a question actually names",
        targets: &[("temporal.rs", "split_list")],
    },
    Probe {
        question: "Stop answering client mistakes with 500",
        targets: &[("serve.rs", "status_for")],
    },
    Probe {
        question: "Give people a way to see their own memory",
        targets: &[
            ("serve.rs", "console_router"),
            ("main.rs", "session_key_seed"),
        ],
    },
    Probe {
        question: "Let \"a week ago\" reach the ranking",
        targets: &[("temporal.rs", "now_rfc3339")],
    },
];

fn probes() -> &'static [Probe] {
    match std::env::var("SCONE_PROBE").as_deref() {
        Ok("commits") => &COMMIT_QUESTIONS,
        _ => &DOC_QUESTIONS,
    }
}

struct Source {
    path: String,
    content: String,
}

fn sources() -> Vec<Source> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut out = Vec::new();
    for crate_dir in ["scone-core", "scone", "scone-bench"] {
        let src = root.join(crate_dir).join("src");
        let mut stack = vec![src];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let content = std::fs::read_to_string(&path).unwrap();
                    let rel = path
                        .strip_prefix(&root)
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    out.push(Source {
                        path: format!("crates/{rel}"),
                        content,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Byte span of `fn name` in `content`: from its declaration line to the
/// next declaration or the end of the file.
fn function_span(content: &str, name: &str) -> Option<(usize, usize)> {
    let needle = format!("fn {name}");
    let mut pos = 0usize;
    let mut start = None;
    for line in content.split_inclusive('\n') {
        let opens = line.trim_start();
        let is_decl = scone_core::chunker::enclosing_declaration(line, 0).is_some();
        if let Some(s) = start {
            if is_decl && !opens.contains(&needle) {
                return Some((s, pos));
            }
        } else if is_decl && opens.contains(&needle) {
            let after = &opens[opens.find(&needle).unwrap() + needle.len()..];
            if after.starts_with(['(', '<']) {
                start = Some(pos);
            }
        }
        pos += line.len();
    }
    start.map(|s| (s, content.len()))
}

/// The source holding `fn name` in the file with basename `file`, and the
/// function's byte span; a target that is not there is a probe bug.
fn locate<'a>(files: &'a [Source], file: &str, name: &str) -> (&'a Source, (usize, usize)) {
    let target = files
        .iter()
        .find(|f| {
            f.path.ends_with(&format!("/{file}")) && function_span(&f.content, name).is_some()
        })
        .unwrap_or_else(|| panic!("no {name} in {file}"));
    let span = function_span(&target.content, name).unwrap();
    (target, span)
}

fn run(contextual: bool) -> (usize, usize, Vec<(usize, Option<usize>)>) {
    let dir = tempfile::tempdir().unwrap();
    let cache = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".scone");
    let mut e = Engine::open(dir.path(), Box::new(OnnxEmbedder::new(&cache).unwrap())).unwrap();
    e.set_contextual_code(contextual);
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let files = sources();
    for f in &files {
        e.import_episode(&space, "file", &f.content, Some(&f.path), None)
            .unwrap();
    }
    let mut at5 = 0;
    let mut at10 = 0;
    let mut ranks = Vec::new();
    for (i, probe) in probes().iter().enumerate() {
        let spans: Vec<(&Source, usize, usize)> = probe
            .targets
            .iter()
            .map(|(file, name)| {
                let (target, (s, end)) = locate(&files, file, name);
                (target, s, end)
            })
            .collect();
        let opts = RecallOpts {
            limit: 10,
            ..Default::default()
        };
        let pack = e.recall(&space, probe.question, &opts).unwrap();
        let rank = pack.items.iter().position(|item| {
            spans.iter().any(|(target, s, end)| {
                item.source.as_deref() == Some(target.path.as_str())
                    && target
                        .content
                        .find(item.text.as_str())
                        .is_some_and(|off| off < *end && off + item.text.len() > *s)
            })
        });
        if rank.is_some_and(|r| r < 5) {
            at5 += 1;
        }
        if rank.is_some() {
            at10 += 1;
        }
        ranks.push((i, rank));
    }
    (at5, at10, ranks)
}

#[test]
#[ignore = "needs the ONNX embedder and a few minutes; run by hand and record the result"]
fn contextual_code_embedding_probe() {
    let files = sources();
    let chunks: usize = files
        .iter()
        .map(|f| chunk_syntax(&f.content, 700, Syntax::Code).len())
        .sum();
    let set = if std::env::var("SCONE_PROBE").as_deref() == Ok("commits") {
        "commits"
    } else {
        "doc"
    };
    println!(
        "files {} chunks {} questions {} (set: {set})",
        files.len(),
        chunks,
        probes().len()
    );
    let (a5, a10, ranks_a) = run(false);
    let (b5, b10, ranks_b) = run(true);
    println!("plain:      R@5 {a5}/20  R@10 {a10}/20");
    println!("contextual: R@5 {b5}/20  R@10 {b10}/20");
    // How much each question borrows from the doc comment it is scored
    // against: the share of the question's words that also appear in
    // that comment. The questions and the comments have the same author,
    // so a high share means the probe is measuring paraphrase of the
    // documentation, not a developer's independent phrasing.
    let words = |s: &str| -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect()
    };
    let mut overlaps = Vec::new();
    for (i, probe) in probes().iter().enumerate() {
        let question = probe.question;
        let name = probe.targets[0].1;
        let ra = ranks_a[i]
            .1
            .map(|r| (r + 1).to_string())
            .unwrap_or_else(|| "-".into());
        let rb = ranks_b[i]
            .1
            .map(|r| (r + 1).to_string())
            .unwrap_or_else(|| "-".into());
        // Against several accepted targets, the largest share counts: the
        // question is only as independent as its closest comment.
        let q = words(question);
        let share = probe
            .targets
            .iter()
            .map(|(file, fn_name)| {
                let (target, (start, _)) = locate(&files, file, fn_name);
                let comment = scone_core::chunker::doc_comment_above(&target.content, start)
                    .unwrap_or_default();
                let c = words(&comment);
                if q.is_empty() {
                    0.0
                } else {
                    q.intersection(&c).count() as f64 / q.len() as f64
                }
            })
            .fold(0.0, f64::max);
        overlaps.push(share);
        println!(
            "  {name:<22} plain {ra:>2}  contextual {rb:>2}  overlap {:.0}%  ({question})",
            share * 100.0
        );
    }
    overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "question/doc-comment word overlap: median {:.0}%, min {:.0}%, max {:.0}%",
        overlaps[overlaps.len() / 2] * 100.0,
        overlaps[0] * 100.0,
        overlaps[overlaps.len() - 1] * 100.0
    );
}
