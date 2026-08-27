#![allow(clippy::unwrap_used)]
use proptest::prelude::*;
use scone_core::chunker::chunk_text;

proptest! {
    /// Spec invariant I1: chunks are contiguous byte spans that reassemble
    /// to exactly the original episode content (memory/bugs.md P-6).
    #[test]
    fn chunks_reassemble_exactly(text in ".{0,4000}", target in 64usize..1024) {
        let spans = chunk_text(&text, target);
        let rebuilt: String = spans.iter().map(|s| &text[s.start..s.end]).collect();
        prop_assert_eq!(&rebuilt, &text);
        for w in spans.windows(2) {
            prop_assert_eq!(w[0].end, w[1].start);
        }
        if let (Some(first), Some(last)) = (spans.first(), spans.last()) {
            prop_assert_eq!(first.start, 0);
            prop_assert_eq!(last.end, text.len());
        }
    }

    #[test]
    fn no_chunk_grossly_oversized(text in "[a-z ]{0,4000}", target in 64usize..512) {
        for s in chunk_text(&text, target) {
            prop_assert!(s.end - s.start <= 2 * target);
        }
    }
}

#[test]
fn empty_text_yields_no_chunks() {
    assert!(chunk_text("", 256).is_empty());
}

#[test]
fn prefers_paragraph_boundaries() {
    let text = "first paragraph here.\n\nsecond paragraph here.";
    let spans = chunk_text(text, 22);
    assert_eq!(
        &text[spans[0].start..spans[0].end],
        "first paragraph here.\n\n"
    );
}

#[test]
fn hard_split_respects_char_boundaries() {
    let text = "é".repeat(500);
    let spans = chunk_text(&text, 64);
    for s in &spans {
        assert!(text.is_char_boundary(s.start) && text.is_char_boundary(s.end));
    }
    let rebuilt: String = spans.iter().map(|s| &text[s.start..s.end]).collect();
    assert_eq!(rebuilt, text);
}
