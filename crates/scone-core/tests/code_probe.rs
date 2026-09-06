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
use scone_core::chunker::{Syntax, chunk_syntax};
use scone_core::embed::OnnxEmbedder;
use scone_core::{Engine, RecallOpts, auth};

/// (question, file basename, function name)
const QUESTIONS: [(&str, &str, &str); 20] = [
    (
        "close the older facts a newly active fact supersedes, or close it if something newer is on record",
        "distill.rs",
        "settle_active_fact",
    ),
    (
        "answer a temporal question by computing over dated episodes, or decline and leave it to the reader",
        "temporal.rs",
        "answer_temporally",
    ),
    (
        "does this line open a declaration, walking past modifiers like pub(crate)",
        "chunker.rs",
        "starts_declaration",
    ),
    (
        "pick prose or code cutting from a source path's extension",
        "chunker.rs",
        "syntax_for",
    ),
    (
        "register another name for an entity",
        "distill.rs",
        "add_entity_alias",
    ),
    ("which episodes taught us a fact", "distill.rs", "facts_why"),
    (
        "close a fact by hand with a reason without deleting it",
        "distill.rs",
        "facts_close",
    ),
    (
        "list the proposed facts waiting for a person",
        "distill.rs",
        "facts_pending",
    ),
    (
        "turn a relative date phrase like a week ago into a window anchored at now",
        "temporal.rs",
        "relative_window",
    ),
    (
        "number of days between two dates",
        "temporal.rs",
        "days_between",
    ),
    (
        "resolve a space name to a scoped space, creating it when allowed",
        "auth.rs",
        "resolve",
    ),
    (
        "recall memory for an agent through the MCP tool with the profile prepended",
        "mcp.rs",
        "memory_recall",
    ),
    (
        "the HTTP handler for recall with query parameters",
        "serve.rs",
        "get_recall",
    ),
    (
        "draw a stratified sample of benchmark items with a seed",
        "lib.rs",
        "stratified_sample",
    ),
    (
        "rebuild the indexes from the database when they are missing or corrupt",
        "lib.rs",
        "doctor_rebuild",
    ),
    ("attach tags to an episode", "tags.rs", "tag_episode"),
    (
        "the profile of a space: identity facts and recent activity",
        "profile.rs",
        "profile",
    ),
    (
        "apply the facts extracted from one episode in one transaction, deduplicating restatements",
        "distill.rs",
        "apply_facts",
    ),
    (
        "read a question as a temporal operator or decide it is not one",
        "temporal.rs",
        "plan",
    ),
    (
        "split a multi-part question into clauses to retrieve for",
        "recall.rs",
        "decompose",
    ),
];

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
    for (i, (question, file, name)) in QUESTIONS.iter().enumerate() {
        let target = files
            .iter()
            .find(|f| {
                f.path.ends_with(&format!("/{file}")) && function_span(&f.content, name).is_some()
            })
            .unwrap_or_else(|| panic!("no {name} in {file}"));
        let (s, end) = function_span(&target.content, name).unwrap();
        let opts = RecallOpts {
            limit: 10,
            ..Default::default()
        };
        let pack = e.recall(&space, question, &opts).unwrap();
        let rank = pack.items.iter().position(|item| {
            item.source.as_deref() == Some(target.path.as_str())
                && target
                    .content
                    .find(item.text.as_str())
                    .is_some_and(|off| off < end && off + item.text.len() > s)
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
    println!(
        "files {} chunks {} questions {}",
        files.len(),
        chunks,
        QUESTIONS.len()
    );
    let (a5, a10, ranks_a) = run(false);
    let (b5, b10, ranks_b) = run(true);
    println!("plain:      R@5 {a5}/20  R@10 {a10}/20");
    println!("contextual: R@5 {b5}/20  R@10 {b10}/20");
    for (i, (question, _, name)) in QUESTIONS.iter().enumerate() {
        let ra = ranks_a[i]
            .1
            .map(|r| (r + 1).to_string())
            .unwrap_or_else(|| "-".into());
        let rb = ranks_b[i]
            .1
            .map(|r| (r + 1).to_string())
            .unwrap_or_else(|| "-".into());
        println!("  {name:<22} plain {ra:>2}  contextual {rb:>2}  ({question})");
    }
}
