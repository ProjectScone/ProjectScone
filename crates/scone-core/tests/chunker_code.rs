#![allow(clippy::unwrap_used)]
//! Code is cut at declarations, not at blank lines. A chunk that starts
//! mid-body reads the same as every other body in the file, so the name
//! it belongs to is the most valuable thing a boundary can preserve.
use scone_core::chunker::{ChunkSpan, Syntax, chunk_syntax, syntax_for};

fn starts_at_declaration(src: &str, spans: &[ChunkSpan]) -> usize {
    spans
        .iter()
        .filter(|s| {
            let line = src[s.start..s.end]
                .lines()
                .next()
                .unwrap_or("")
                .trim_start();
            let word = line
                .split(|c: char| c.is_whitespace() || c == '(')
                .next()
                .unwrap_or("");
            !line.starts_with("//")
                && [
                    "fn", "pub", "struct", "enum", "impl", "trait", "const", "type", "async",
                    "def", "class", "func", "function",
                ]
                .contains(&word)
        })
        .count()
}

#[test]
fn code_mode_aligns_more_chunks_to_declarations_than_prose_mode() {
    let src = std::fs::read_to_string("src/distill.rs").unwrap();
    let prose = chunk_syntax(&src, 1000, Syntax::Prose);
    let code = chunk_syntax(&src, 1000, Syntax::Code);
    let prose_hits = starts_at_declaration(&src, &prose);
    let code_hits = starts_at_declaration(&src, &code);
    assert!(
        code_hits > prose_hits,
        "code mode must beat prose on its own file: {code_hits} vs {prose_hits}"
    );
}

/// The whole point is losing nothing. Spans must still tile the input
/// exactly in code mode, or provenance is gone.
#[test]
fn code_chunks_still_reassemble_exactly() {
    for file in ["src/recall.rs", "src/ingest.rs", "src/chunker.rs"] {
        let src = std::fs::read_to_string(file).unwrap();
        let spans = chunk_syntax(&src, 700, Syntax::Code);
        let mut at = 0usize;
        let mut rebuilt = String::new();
        for s in &spans {
            assert_eq!(s.start, at, "{file}: gap or overlap at {at}");
            rebuilt.push_str(&src[s.start..s.end]);
            at = s.end;
        }
        assert_eq!(at, src.len(), "{file}: spans stop short");
        assert_eq!(rebuilt, src, "{file}: content changed");
    }
}

#[test]
fn declarations_are_recognized_across_languages() {
    let cases = [
        ("fn main() {", true),
        ("    pub fn day(&self) -> &str {", true),
        ("pub(crate) fn open(path: &Path) {", true),
        ("pub async fn recall(", true),
        ("def handler(request):", true),
        ("class Memory:", true),
        ("export function search(q) {", true),
        ("func (s *Server) Handle() {", true),
        ("public static void main(String[] a) {", true),
        ("impl Connector for Notion {", true),
        ("// fn in a comment", false),
        ("    let spans = chunk_text(content);", false),
        ("", false),
        ("        }", false),
    ];
    for (line, expect) in cases {
        let src = format!("filler line\n{line}\nbody\n");
        let at = src.find(line).filter(|_| !line.is_empty());
        let spans = chunk_syntax(&src, 8, Syntax::Code);
        // The declaration must begin a span, indentation included.
        let cut_before = at.is_some_and(|at| at > 0 && spans.iter().any(|s| s.start == at));
        assert_eq!(
            cut_before,
            expect,
            "{line:?} should {} start a chunk",
            if expect { "" } else { "not" }
        );
    }
}

#[test]
fn syntax_is_chosen_by_source_extension() {
    assert_eq!(syntax_for(Some("src/main.rs")), Syntax::Code);
    assert_eq!(syntax_for(Some("/a/b/handler.PY")), Syntax::Code);
    assert_eq!(syntax_for(Some("notes/meeting.md")), Syntax::Prose);
    assert_eq!(syntax_for(Some("https://example.com/post")), Syntax::Prose);
    assert_eq!(syntax_for(None), Syntax::Prose);
    // A note that merely mentions code is still prose.
    assert_eq!(syntax_for(Some("README")), Syntax::Prose);
}
