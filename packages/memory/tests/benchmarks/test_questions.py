"""A question set is only as true as its quotes: what is kept, what is dropped, and what it measures."""
from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench import questions as module
from scone_memory.bench.questions import QuestionSet, chunks_in, measure, store_corpus, write_questions
from scone_memory.core.errors import InvalidInput
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.runtime import cli

HARBOUR = ("The harbour at Vellmar closes to sailing boats every November. "
           "Its lighthouse was rebuilt in 1904 after a storm took the first one. "
           "Pilots board arriving ships two miles out, at the red buoy.")
ORCHARD = ("The orchard on the ridge grows only Bramley apples. "
           "Picking starts in the last week of September and ends before the first frost. "
           "The press in the barn makes two thousand litres of juice a season.")


async def corpus():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000).open()
    await engine.remember("s", HARBOUR, kind="file", source="harbour.md")
    await engine.remember("s", ORCHARD, kind="file", source="orchard.md")
    return engine


def reply(*pairs):
    return json.dumps([{"question": q, "quote": a} for q, a in pairs])


async def test_questions_are_kept_only_with_a_quote_that_is_in_the_chunk():
    engine = await corpus()
    try:
        model = FakeChat([
            reply(("When does the harbour close to sailing boats?", "The harbour at Vellmar closes to sailing boats every November."),
                  ("What happened to the first lighthouse?", "A storm destroyed it in 1903.")),
            "```json\n" + reply(("Which apples grow on the ridge?", "The orchard on the ridge grows  only Bramley apples.")) + "\n```",
        ])
        written = await write_questions(engine, "s", model, corpus="two notes", model_name="fake", per_chunk=2)
        assert [q.question for q in written.questions] == ["When does the harbour close to sailing boats?",
                                                           "Which apples grow on the ridge?"]
        assert written.questions[0].quote == "The harbour at Vellmar closes to sailing boats every November."
        assert written.questions[0].source == "harbour.md" and written.questions[1].source == "orchard.md"
        assert (written.dropped_unquoted, written.dropped_unparsed, written.calls_failed) == (1, 0, 0), "the invented quote is dropped"
        assert (written.chunks_total, written.chunks_asked, written.model, written.corpus) == (2, 2, "fake", "two notes")
        assert "Passage:\n" + HARBOUR in model.calls[0][1] and "2 question(s)" in model.calls[0][1]
        assert "dropped 1 whose quote was not in the chunk" in written.text()
    finally:
        await engine.close()


async def test_what_the_model_did_not_write_as_asked_is_counted_and_never_a_question():
    engine = await corpus()
    try:
        model = FakeChat(["Here are some questions: 1. When? 2. Where?", ChatError("model down")])
        written = await write_questions(engine, "s", model, per_chunk=1)
        assert written.questions == ()
        assert (written.dropped_unparsed, written.calls_failed, written.dropped_unquoted) == (1, 1, 0)
        model = FakeChat([json.dumps([{"question": "Q?", "quote": 5}]), json.dumps({"question": "Q?", "quote": "x"})])
        again = await write_questions(engine, "s", model, per_chunk=1)
        assert again.questions == () and again.dropped_unparsed == 2, "a wrong shape is not a question"
        with pytest.raises(InvalidInput, match="per_chunk"):
            await write_questions(engine, "s", FakeChat(), per_chunk=0)
    finally:
        await engine.close()


async def test_a_large_corpus_is_sampled_by_seed_and_the_set_says_how_many_there_were(monkeypatch):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000).open()
    try:
        for number in range(6):
            await engine.remember("s", f"Note number {number} says that the bell rings at {number} o'clock.", source=f"n{number}.md")
        model = FakeChat([reply((f"Q{n}?", "the bell rings at")) for n in range(3)])
        first = await write_questions(engine, "s", model, per_chunk=1, max_chunks=3, seed=7)
        assert (first.chunks_total, first.chunks_asked) == (6, 3) and len(first.questions) == 3
        model = FakeChat([reply((f"Q{n}?", "the bell rings at")) for n in range(3)])
        second = await write_questions(engine, "s", model, per_chunk=1, max_chunks=3, seed=7)
        assert [q.episode_id for q in second.questions] == [q.episode_id for q in first.questions], "the same seed, the same chunks"
        monkeypatch.setattr(module, "MAX_CHUNK_BYTES", 30)
        model = FakeChat([])
        bounded = await write_questions(engine, "s", model, per_chunk=1)
        assert bounded.skipped_long == 6 and bounded.chunks_asked == 0 and model.calls == [], "a long chunk is not shown"
    finally:
        await engine.close()


async def test_a_set_round_trips_through_a_file_and_refuses_another_version(tmp_path):
    engine = await corpus()
    try:
        written = await write_questions(engine, "s", FakeChat([reply(("Where do pilots board?", "Pilots board arriving ships two miles out, at the red buoy.")), reply()]), per_chunk=1)
        written.save(tmp_path / "set.json")
        loaded = QuestionSet.load(tmp_path / "set.json")
        assert loaded == written
        payload = written.as_payload()
        payload["version"] = "questions-v0"
        (tmp_path / "old.json").write_text(json.dumps(payload))
        with pytest.raises(InvalidInput, match="questions-v0"):
            QuestionSet.load(tmp_path / "old.json")
        with pytest.raises(InvalidInput, match="cannot read"):
            QuestionSet.load(tmp_path / "missing.json")
    finally:
        await engine.close()


async def test_measurement_scores_a_passage_by_the_quote_it_holds():
    engine = await corpus()
    try:
        written = await write_questions(engine, "s", FakeChat([
            reply(("When does the harbour close?", "The harbour at Vellmar closes to sailing boats every November.")),
            reply(("How much juice does the press make?", "The press in the barn makes two thousand litres of juice a season.")),
        ]), per_chunk=1)
        report = await measure(engine, "s", written, ks=(1, 2))
        assert report.questions == 2 and report.quote_at == {1: 1.0, 2: 1.0} and report.mrr == 1.0 and report.unfound == 0
        assert report.source_at == {1: 1.0, 2: 1.0} and report.with_source == 2
        elsewhere = QuestionSet.from_payload({**written.as_payload(), "questions": [
            {"question": "What colour is the buoy?", "quote": "the red buoy", "source": "harbour.md", "episode_id": 1},
            {"question": "Anything about trains?", "quote": "the night train to Kassel", "source": None, "episode_id": 9}]})
        report = await measure(engine, "s", elsewhere, ks=(1,))
        assert report.quote_at == {1: 0.5} and report.unfound == 1 and report.with_source == 1
        assert "1 unfound at 1" in report.text()
        with pytest.raises(InvalidInput, match="limit"):
            await measure(engine, "s", written, ks=(5,), limit=2)
        assert (await measure(engine, "s", QuestionSet.from_payload({**written.as_payload(), "questions": []}))).text().startswith("no questions")
    finally:
        await engine.close()


async def test_a_corpus_root_is_stored_in_a_settled_order_with_its_sources(tmp_path):
    (tmp_path / "b.md").write_text(ORCHARD, encoding="utf-8")
    (tmp_path / "a.txt").write_text(HARBOUR, encoding="utf-8")
    (tmp_path / "skip.py").write_text("print('not a document')", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        stored = await store_corpus(engine, "s", tmp_path)
        assert stored == {"files": 2, "files_total": 3, "files_cut": 0, "pdf_unread": 0}
        assert [source for _, source, _, _ in await chunks_in(engine, "s")] == ["a.txt", "b.md"]
    finally:
        await engine.close()


def test_the_command_writes_with_a_model_and_measures_without_one(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "harbour.md").write_text(HARBOUR, encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench-questions", str(root), "--set", str(tmp_path / "set.json"), "--write"],
                    env={}, stdin=io.StringIO(""), out=out)
    assert code == 2, "writing needs a configured model"
    from scone_memory.runtime import config

    monkeypatch.setattr(config, "build_chat", lambda settings: FakeChat([
        reply(("When does the harbour close?", "The harbour at Vellmar closes to sailing boats every November."))]))
    out = io.StringIO()
    code = cli.main(["bench-questions", str(root), "--set", str(tmp_path / "set.json"), "--write", "--per-chunk", "1"],
                    env={}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    assert "1 question(s) from 1 of 1 chunk(s)" in out.getvalue(), out.getvalue()
    out = io.StringIO()
    code = cli.main(["bench-questions", str(root), "--set", str(tmp_path / "set.json"), "--k", "1,3", "--json"],
                    env={}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    payload = json.loads(out.getvalue())
    assert payload["quote_at"] == {"1": 1.0, "3": 1.0} and payload["stored"]["files"] == 1
    out = io.StringIO()
    assert cli.main(["bench-questions", str(tmp_path / "nowhere"), "--set", "x.json"], env={}, stdin=io.StringIO(""), out=out) == 2
