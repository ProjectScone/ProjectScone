//! Structure-aware chunking over immutable episode content.
//!
//! Chunks are contiguous byte spans covering the whole text (spec invariant
//! I1): they reassemble to exactly the original, so provenance can never be
//! destroyed in the pipeline (memory/bugs.md P-6). Cuts prefer blank lines
//! and markdown heading starts; a span that would exceed 2x the target is
//! hard-split at char boundaries.

/// A contiguous byte range of an episode's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSpan {
    pub start: usize,
    pub end: usize,
}

/// What kind of text is being cut. Prose and code want different
/// boundaries: a paragraph break means nothing in a function body, and
/// a declaration line means nothing in an essay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Syntax {
    #[default]
    Prose,
    Code,
}

/// Modifiers that can precede a declaration keyword in the languages
/// people actually store code in.
const MODIFIERS: [&str; 12] = [
    "pub",
    "export",
    "async",
    "static",
    "final",
    "public",
    "private",
    "protected",
    "unsafe",
    "extern",
    "default",
    "abstract",
];

/// Words that open a block without naming anything. The C-family rule
/// below would otherwise treat every `if (x) {` as a declaration.
const CONTROL: [&str; 14] = [
    "if", "for", "while", "switch", "match", "catch", "else", "do", "try", "loop", "elif",
    "except", "with", "return",
];

/// Keywords that open a named, addressable thing.
const DECLARATIONS: [&str; 14] = [
    "fn",
    "def",
    "class",
    "struct",
    "enum",
    "impl",
    "trait",
    "func",
    "function",
    "interface",
    "type",
    "mod",
    "module",
    "package",
];

/// Does this line open a declaration? Chunks that start here keep their
/// own name; chunks that start mid-body are orphaned code that reads
/// the same from every function in the file.
fn starts_declaration(line: &str) -> bool {
    let mut rest = line.trim_start();
    // Walk past any stack of modifiers, including pub(crate).
    loop {
        let word = rest
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("");
        if word.is_empty() {
            return false;
        }
        if DECLARATIONS.contains(&word) {
            return true;
        }
        if !MODIFIERS.contains(&word) {
            // C, C++, C#, Java: the return type stands where the keyword
            // would be, so the shape has to carry the signal instead.
            // `String getName() {` counts; `if (x) {` and `} else {` do not.
            let looks_callable = word.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                && !CONTROL.contains(&word)
                && rest.trim_end().ends_with('{')
                && rest
                    .split_once('(')
                    .is_some_and(|(before, _)| !before.contains('='));
            return looks_callable;
        }
        let after = &rest[word.len()..];
        // pub(crate) and friends: skip the parenthesized scope.
        let after = match after.strip_prefix('(') {
            Some(inner) => match inner.find(')') {
                Some(i) => &inner[i + 1..],
                None => return false,
            },
            None => after,
        };
        let trimmed = after.trim_start();
        if trimmed.len() == after.len() && !after.is_empty() {
            return false;
        }
        rest = trimmed;
    }
}

/// Extensions whose contents are cut at declarations rather than at
/// blank lines. Judged by the source path, so a note about code is
/// still prose and a pulled file is still whatever it is.
const CODE_EXTENSIONS: [&str; 24] = [
    "rs", "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "go", "java", "c", "h", "cc", "cpp", "hpp",
    "cs", "rb", "php", "swift", "kt", "scala", "lua", "zig", "dart",
];

/// Pick a cutting strategy from a source path.
pub fn syntax_for(source: Option<&str>) -> Syntax {
    let Some(source) = source else {
        return Syntax::Prose;
    };
    let ext = source
        .rsplit('/')
        .next()
        .and_then(|name| name.rsplit_once('.'))
        .map(|(_, ext)| ext.to_ascii_lowercase());
    match ext {
        Some(ext) if CODE_EXTENSIONS.contains(&ext.as_str()) => Syntax::Code,
        _ => Syntax::Prose,
    }
}

pub fn chunk_text(text: &str, target_bytes: usize) -> Vec<ChunkSpan> {
    chunk_syntax(text, target_bytes, Syntax::Prose)
}

pub fn chunk_syntax(text: &str, target_bytes: usize, syntax: Syntax) -> Vec<ChunkSpan> {
    let target = target_bytes.max(1);
    if text.is_empty() {
        return Vec::new();
    }

    // Pass 1: cut at preferred boundaries once the target is reached.
    let mut soft_spans = Vec::new();
    let mut chunk_start = 0usize;
    let mut last_soft: Option<usize> = None;
    let mut last_decl: Option<usize> = None;
    let mut pos = 0usize;
    for line in text.split_inclusive('\n') {
        let line_start = pos;
        pos += line.len();
        // A heading line starts a new logical section: cut before it.
        if line.starts_with('#') && line_start > chunk_start {
            last_soft = Some(line_start);
        }
        // In code, a declaration is the heading: cut before it so the
        // chunk that follows carries its own signature. Tracked apart
        // from blank lines, because code is full of blank lines and the
        // most recent one is usually mid-function.
        if syntax == Syntax::Code && line_start > chunk_start && starts_declaration(line) {
            // Cut here as soon as the chunk holds enough to be useful,
            // rather than waiting to pass the target. A declaration
            // reached later is behind the next chunk's start, so
            // waiting is how bodies end up orphaned from their names.
            if line_start - chunk_start >= target / 4 {
                soft_spans.push(ChunkSpan {
                    start: chunk_start,
                    end: line_start,
                });
                chunk_start = line_start;
                last_soft = None;
                last_decl = None;
                continue;
            }
            last_decl = Some(line_start);
        }
        // A blank line ends a paragraph: cut after it.
        if line.trim().is_empty() {
            last_soft = Some(pos);
        }
        // Cut only at preferred boundaries; text with none is handled by
        // the 2x-target hard-split below.
        if pos - chunk_start >= target {
            // A declaration wins over a blank line, but only once the
            // chunk has enough in it to be worth keeping; otherwise a
            // run of one-line declarations would shred the file.
            let decl_cut = last_decl.filter(|s| *s > chunk_start && s - chunk_start >= target / 2);
            let cut = decl_cut.or_else(|| last_soft.filter(|s| *s > chunk_start));
            if let Some(cut) = cut {
                soft_spans.push(ChunkSpan {
                    start: chunk_start,
                    end: cut,
                });
                chunk_start = cut;
                last_soft = None;
                last_decl = None;
            }
        }
    }
    if chunk_start < text.len() {
        soft_spans.push(ChunkSpan {
            start: chunk_start,
            end: text.len(),
        });
    }

    // Pass 2: hard-split anything still over 2x target at char boundaries.
    let mut spans = Vec::with_capacity(soft_spans.len());
    for span in soft_spans {
        if span.end - span.start <= 2 * target {
            spans.push(span);
            continue;
        }
        let mut piece_start = span.start;
        let mut last_boundary = span.start;
        for (off, ch) in text[span.start..span.end].char_indices() {
            let abs = span.start + off;
            if abs - piece_start >= target {
                spans.push(ChunkSpan {
                    start: piece_start,
                    end: abs,
                });
                piece_start = abs;
            }
            last_boundary = abs + ch.len_utf8();
        }
        if piece_start < last_boundary {
            spans.push(ChunkSpan {
                start: piece_start,
                end: span.end,
            });
        }
    }
    spans
}
