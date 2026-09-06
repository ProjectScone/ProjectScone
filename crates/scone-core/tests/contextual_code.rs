#![allow(clippy::unwrap_used)]
//! Contextual embedding for code: a chunk cut from the middle of a
//! function is embedded with the file name and the declaration it sits
//! under in front, while the stored span stays the raw bytes. Off by
//! default; prose never gets a prefix.
use std::sync::{Arc, Mutex};

use scone_core::chunker::{ChunkSpan, contextual_code_text, enclosing_declaration};
use scone_core::embed::EmbeddingProvider;
use scone_core::{Engine, auth};

/// Records what it was asked to embed; vectors are hashed so the index
/// accepts them.
struct Recording {
    seen: Arc<Mutex<Vec<String>>>,
    inner: scone_core::embed::HashEmbedder,
}

impl EmbeddingProvider for Recording {
    fn id(&self) -> &str {
        "hash-64"
    }
    fn dim(&self) -> usize {
        64
    }
    fn embed(&self, texts: &[&str]) -> scone_core::Result<Vec<Vec<f32>>> {
        self.seen
            .lock()
            .unwrap()
            .extend(texts.iter().map(|t| t.to_string()));
        self.inner.embed(texts)
    }
}

const SOURCE: &str = "use std::fmt;\n\n\
pub fn first(a: u32) -> u32 {\n    let b = a + 1;\n    b * 2\n}\n\n\
pub fn second(x: &str) -> String {\n    let y = x.trim();\n    y.to_uppercase()\n}\n";

#[test]
fn enclosing_declaration_is_the_nearest_one_above() {
    let first_body = SOURCE.find("let b").unwrap();
    assert_eq!(
        enclosing_declaration(SOURCE, first_body),
        Some("pub fn first(a: u32) -> u32")
    );
    let second_body = SOURCE.find("y.to_uppercase").unwrap();
    assert_eq!(
        enclosing_declaration(SOURCE, second_body),
        Some("pub fn second(x: &str) -> String")
    );
    assert_eq!(
        enclosing_declaration(SOURCE, 0),
        None,
        "the file header sits under nothing"
    );
    assert_eq!(enclosing_declaration("just a script line\n", 5), None);
    assert_eq!(
        enclosing_declaration(SOURCE, usize::MAX),
        Some("pub fn second(x: &str) -> String")
    );
}

#[test]
fn contextual_text_names_the_file_and_the_declaration() {
    let start = SOURCE.find("    let y").unwrap();
    let span = ChunkSpan {
        start,
        end: SOURCE.len(),
    };
    assert_eq!(
        contextual_code_text(Some("crates/x/src/lib.rs"), SOURCE, span),
        format!(
            "lib.rs | pub fn second(x: &str) -> String\n{}",
            &SOURCE[start..]
        )
    );
    assert_eq!(
        contextual_code_text(None, SOURCE, span),
        format!("pub fn second(x: &str) -> String\n{}", &SOURCE[start..])
    );
    let head = ChunkSpan { start: 0, end: 14 };
    assert_eq!(
        contextual_code_text(Some("a/b.rs"), SOURCE, head),
        format!("b.rs\n{}", &SOURCE[..14])
    );
    assert_eq!(
        contextual_code_text(None, SOURCE, head),
        SOURCE[..14].to_owned()
    );
}

fn engine_with(seen: Arc<Mutex<Vec<String>>>, dir: &std::path::Path) -> Engine {
    let embedder = Recording {
        seen,
        inner: scone_core::embed::HashEmbedder::new(64),
    };
    Engine::open(dir, Box::new(embedder)).unwrap()
}

#[test]
fn code_is_embedded_with_context_only_when_asked_and_only_for_code() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine_with(seen.clone(), dir.path());
    e.set_chunk_target(64);
    let space = auth::resolve(&mut e, "default", true).unwrap();

    // Off (the default): the vector sees the raw chunk, code or not.
    e.import_episode(&space, "file", SOURCE, Some("src/lib.rs"), None)
        .unwrap();
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|t| !t.starts_with("lib.rs")),
        "{:?}",
        seen.lock().unwrap()
    );
    seen.lock().unwrap().clear();

    e.set_contextual_code(true);
    let content = format!("{SOURCE}\n"); // not a duplicate of the first import
    let (episode_id, _) = e
        .import_episode(&space, "file", &content, Some("src/lib.rs"), None)
        .unwrap();
    let embedded = seen.lock().unwrap().clone();
    assert!(embedded.len() >= 2, "several chunks: {embedded:?}");
    assert!(
        embedded.iter().all(|t| t.starts_with("lib.rs")),
        "{embedded:?}"
    );
    assert!(
        embedded
            .iter()
            .any(|t| t.starts_with("lib.rs | pub fn second(x: &str) -> String\n")),
        "a mid-function chunk carries its declaration: {embedded:?}"
    );
    // The stored bytes are the raw span: recall returns the code, not the prefix.
    let pack = e
        .recall(&space, "y.to_uppercase", &scone_core::RecallOpts::default())
        .unwrap();
    let mine: Vec<_> = pack
        .items
        .iter()
        .filter(|i| i.episode_id == episode_id)
        .collect();
    assert!(!mine.is_empty());
    assert!(
        mine.iter()
            .all(|i| !i.text.is_empty() && content.contains(i.text.as_str())),
        "every returned chunk is a raw span of the source: {mine:?}"
    );
    seen.lock().unwrap().clear();

    // Prose stays prose even with the switch on.
    e.import_episode(
        &space,
        "note",
        "a paragraph about lib.rs and its second function",
        Some("notes/lib.md"),
        None,
    )
    .unwrap();
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|t| !t.starts_with("lib.md")),
        "{:?}",
        seen.lock().unwrap()
    );
}
