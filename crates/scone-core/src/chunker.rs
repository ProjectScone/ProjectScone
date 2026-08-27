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

pub fn chunk_text(text: &str, target_bytes: usize) -> Vec<ChunkSpan> {
    let target = target_bytes.max(1);
    if text.is_empty() {
        return Vec::new();
    }

    // Pass 1: cut at preferred boundaries once the target is reached.
    let mut soft_spans = Vec::new();
    let mut chunk_start = 0usize;
    let mut last_soft: Option<usize> = None;
    let mut pos = 0usize;
    for line in text.split_inclusive('\n') {
        let line_start = pos;
        pos += line.len();
        // A heading line starts a new logical section: cut before it.
        if line.starts_with('#') && line_start > chunk_start {
            last_soft = Some(line_start);
        }
        // A blank line ends a paragraph: cut after it.
        if line.trim().is_empty() {
            last_soft = Some(pos);
        }
        // Cut only at preferred boundaries; text with none is handled by
        // the 2x-target hard-split below.
        if pos - chunk_start >= target
            && let Some(cut) = last_soft.filter(|s| *s > chunk_start)
        {
            soft_spans.push(ChunkSpan {
                start: chunk_start,
                end: cut,
            });
            chunk_start = cut;
            last_soft = None;
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
