"""Code-point spans become byte spans exactly as a character-by-character table says.

The conversion runs for every record that is not plain ASCII, so it encodes
the stretches between the spans' ends rather than every character alone.
These tests hold it to the table built one character at a time: on every
width of UTF-8, on spans in any order, empty or touching either end, on this
package's own documents, and on spans that are not spans of the text at all.
"""

from __future__ import annotations

import random

import pytest

from scone_memory.ingestion.chunker import Span, byte_spans, chunk_spans

from ..paths import PACKAGE_ROOT


def one_character_at_a_time(content: str, spans: list[Span]) -> list[Span]:
    offsets = [0]
    for ch in content:
        offsets.append(offsets[-1] + len(ch.encode()))
    return [Span(offsets[s.start], offsets[s.end]) for s in spans]


TEXTS = [
    "caf\u00e9 cr\u00e8me br\u00fbl\u00e9e",
    "\u6771\u4eac\u30bf\u30ef\u30fc\u306f\u6771\u4eac\u306b\u3042\u308b\u3002",
    "emoji \U0001f600 and \U0001f9ea in a line\r\nnext",
    "e\u0301 combining marks a\u0308",
    "ascii start, then \u00fc at the end \u00fc",
]


def test_every_width_of_utf8_converts_as_the_table_says():
    chooser = random.Random(5)
    for text in TEXTS:
        size = len(text)
        spans = [Span(0, size), Span(0, 0), Span(size, size), Span(3, 7), Span(1, 2), Span(2, size)]
        spans += [Span(*sorted((chooser.randint(0, size), chooser.randint(0, size)))) for _ in range(20)]
        chooser.shuffle(spans)
        assert byte_spans(text, spans) == one_character_at_a_time(text, spans)
        for span, converted in zip(spans, byte_spans(text, spans)):
            assert text.encode()[converted.start:converted.end].decode() == text[span.start:span.end]


def test_this_packages_documents_convert_as_the_table_says():
    documents = sorted((PACKAGE_ROOT / "docs").rglob("*.md"))
    unicode = [text for path in documents if not (text := path.read_text(encoding="utf-8")).isascii()]
    assert len(unicode) >= 5
    for text in unicode:
        spans = chunk_spans(text, 300)
        assert byte_spans(text, spans) == one_character_at_a_time(text, spans)


def test_an_ascii_text_keeps_the_spans_it_was_given():
    spans = [Span(0, 3), Span(3, 5)]
    assert byte_spans("plain", spans) is spans


def test_a_span_past_the_end_is_refused_as_before():
    with pytest.raises(IndexError):
        one_character_at_a_time("caf\u00e9", [Span(0, 9)])
    with pytest.raises(IndexError):
        byte_spans("caf\u00e9", [Span(0, 9)])


def test_a_negative_offset_counts_from_the_end_as_it_always_did():
    text = "caf\u00e9 cr\u00e8me"
    spans = [Span(-3, -1), Span(0, -2)]
    assert byte_spans(text, spans) == one_character_at_a_time(text, spans)


def test_text_that_cannot_be_utf8_is_refused_wherever_the_bad_character_is():
    for text in ("\ud800 caf\u00e9", "caf\u00e9 \ud800"):
        with pytest.raises(UnicodeEncodeError):
            byte_spans(text, [Span(0, 1)])
