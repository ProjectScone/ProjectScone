"""Where in a returned passage the question's words occur.

A recalled passage says which lanes found it and how it ranked; it does
not say where the words that matched are. A reader, or a page drawing
the passage, has to tokenise it again to show that -- and gets it wrong
the moment their tokeniser differs from the lexical lane's, which folds
case, normalises width and splits scripts without spaces into
characters. `highlights` gives the spans, in code points of the passage
exactly as returned, of every word whose tokens include a query term, by
the same tokeniser the lexical lane uses. It is bounded per passage and
says when the bound bit, and it is computed last, after every stage that
changes a passage's text, so the spans always point into what the reader
has.
"""
from __future__ import annotations

import pytest

from scone_memory.core.models import RecallItem
from scone_memory.retrieval.highlights import MAX_HIGHLIGHTS, highlights
from scone_memory.retrieval.lexical import tokenize


def item(text, chunk_id=1):
    return RecallItem(chunk_id=chunk_id, episode_id=1, text=text, score=1.0, created_at="2026-01-01T00:00:00Z")


def spans(found):
    return [(span.start, span.end) for span in found.spans]


def test_each_matching_word_is_located_in_the_text_as_returned():
    [found] = highlights([item("The harbour Crane was repainted; cranes rust.")], "crane repaint")
    text = "The harbour Crane was repainted; cranes rust."
    assert [text[s:e] for s, e in spans(found)] == ["Crane"]
    assert found.terms == ["crane", "repaint"] and found.truncated is False


def test_case_and_width_fold_as_the_lexical_lane_folds_them():
    text = "ＺÜRICH and zürich and Zurich"
    [found] = highlights([item(text)], "zürich")
    assert [text[s:e] for s, e in spans(found)] == ["ＺÜRICH", "zürich"]


def test_a_possessive_matches_its_word():
    text = "Alves's notebook"
    [found] = highlights([item(text)], "alves")
    assert [text[s:e] for s, e in spans(found)] == ["Alves's"]


def test_a_script_without_spaces_is_located_by_its_characters():
    text = "東京タワーの夜景"
    [found] = highlights([item(text)], "東京")
    assert [text[s:e] for s, e in spans(found)] == ["東京"]


def test_stopwords_in_the_question_highlight_nothing():
    [found] = highlights([item("the crane and the harbour")], "the and")
    assert found.spans == [] and found.terms == []


def test_every_span_names_a_word_the_lexical_lane_would_match():
    text = "Workers' crane logs: crane-2 failed, CRANE 3 passed; the cranes' manual."
    query = "crane logs"
    [found] = highlights([item(text)], query)
    terms = set(tokenize(query))
    assert found.spans
    for s, e in spans(found):
        assert set(tokenize(text[s:e])) & terms, text[s:e]


def test_the_bound_is_per_passage_and_disclosed():
    text = " ".join(["crane"] * (MAX_HIGHLIGHTS + 5))
    [found] = highlights([item(text)], "crane")
    assert len(found.spans) == MAX_HIGHLIGHTS and found.truncated is True and found.total == MAX_HIGHLIGHTS + 5


def test_one_result_per_item_in_order():
    found = highlights([item("crane", 1), item("nothing here", 2)], "crane")
    assert [f.chunk_id for f in found] == [1, 2] and spans(found[1]) == []


async def test_over_http_it_is_opt_in_and_applied_after_the_window():
    from httpx import ASGITransport, AsyncClient

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api import create_app

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=120).open()
    try:
        await engine.remember("s", "Intro text about ships and ports. " * 6 + "The harbour crane was repainted in May. " + "Closing lines. " * 6)
        async with AsyncClient(transport=ASGITransport(app=create_app(engine, {"k": "s"})), base_url="http://fixture") as client:
            auth = {"authorization": "Bearer k"}
            plain = (await client.get("/v1/recall", params={"q": "crane repainted"}, headers=auth)).json()
            assert "highlights" not in plain
            wide = (await client.get("/v1/recall", params={"q": "crane repainted", "highlight": "true", "window": 200}, headers=auth)).json()
            assert len(wide["highlights"]) == len(wide["items"])
            for passage, marks in zip(wide["items"], wide["highlights"]):
                assert marks["chunk_id"] == passage["chunk_id"]
                for span in marks["spans"]:
                    assert set(tokenize(passage["text"][span["start"]:span["end"]])) & set(tokenize("crane repainted"))
            assert any(marks["spans"] for marks in wide["highlights"])
    finally:
        await engine.close()


async def test_on_the_command_line():
    import io
    import json

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.runtime.cli import build_parser, run

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        await engine.remember("default", "The harbour crane was repainted in May.")
        out = io.StringIO()
        code = await run(build_parser().parse_args(["--json", "recall", "crane", "--highlight"]), engine, io.StringIO(""), out)
        body = json.loads(out.getvalue())
        assert code == 0 and body["highlights"][0]["spans"]
    finally:
        await engine.close()


async def test_on_the_command_line_without_json_it_names_the_matched_words():
    import io

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.runtime.cli import build_parser, run

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        await engine.remember("default", "The harbour Crane was repainted in May.")
        out = io.StringIO()
        code = await run(build_parser().parse_args(["recall", "crane repainted", "--highlight"]), engine, io.StringIO(""), out)
        assert code == 0 and "matched: Crane, repainted" in out.getvalue()
    finally:
        await engine.close()
