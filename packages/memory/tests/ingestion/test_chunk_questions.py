"""The question lane: a chunk found by the questions it answers, never by words it does not hold.

A model writes, for each chunk, questions the chunk answers, each with the
sentence that answers it copied from the chunk. A question whose sentence
is not in the chunk is dropped and counted. The kept questions go into an
index of their own beside the chunk's text, so a query phrased like a
question finds the chunk even when it shares no word with it; what comes
back is the chunk's own text. The words a chunk is under are another
lane's and a pass never touches them. Forgetting the episode takes its
questions with it, even while a pass is waiting on the model, and a store
without the index says so.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.backends import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import TextFilter
from scone_memory.ingestion import chunk_questions as module
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.runtime.cli import build_parser, run

HARBOUR = ("The harbour at Vellmar closes to sailing boats every November. "
           "Its lighthouse was rebuilt in 1904 after a storm took the first one. "
           "Pilots board arriving ships two miles out, at the red buoy.")
ORCHARD = ("The orchard on the ridge grows only Bramley apples. "
           "Picking starts in the last week of September and ends before the first frost. "
           "The press in the barn makes two thousand litres of juice a season.")
#: Shares no word with any stored chunk: only the question lane can find the harbour by it.
ASKED = "winter mooring ban"
QUESTION = "Is there a winter mooring ban for yachts?"
QUOTE = "The harbour at Vellmar closes to sailing boats every November."


def reply(*pairs: tuple[str, str]) -> str:
    return json.dumps([{"question": q, "quote": a} for q, a in pairs])


@pytest.fixture(params=["memory", "sqlite"])
async def stores(request, tmp_path):
    documents = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(str(tmp_path / "s.db"))
    yield documents
    close = getattr(documents, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result


async def corpus(documents, *, question_lane: bool = True, context_lane: bool = False,
                 vectors: InMemoryVectorIndex | None = None) -> MemoryEngine:
    engine = await MemoryEngine(documents, vectors or InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000,
                                question_lane=question_lane, context_lane=context_lane, events=InMemoryEventLog()).open()
    await engine.remember("s", HARBOUR, kind="file", source="harbour.md")
    await engine.remember("s", ORCHARD, kind="file", source="orchard.md")
    return engine


async def test_a_query_like_a_kept_question_finds_the_chunk_and_the_passage_is_the_chunk(stores):
    vectors = InMemoryVectorIndex()
    engine = await corpus(stores, vectors=vectors)
    model = FakeChat([reply((QUESTION, QUOTE)), reply(("When does picking start?", "Picking starts in the last week of September"
                                                                                  " and ends before the first frost."))])
    report = await engine.build_chunk_questions("s", model, per_chunk=2, model_name="fake-3b")
    assert (report.chunks_total, report.chunks_asked, report.model_calls, report.chunks_indexed) == (2, 2, 2, 2)
    assert report.questions == 2 and report.kept_lane and report.model == "fake-3b"
    assert [kept.questions for kept in report.kept] == [(QUESTION,), ("When does picking start?",)]
    assert "Passage:" in model.calls[0][1] and HARBOUR in model.calls[0][1] and model.calls[0][0] == module.SYSTEM

    found = await engine.recall("s", ASKED, limit=2)
    assert found.items[0].text == HARBOUR, "the passage returned is the chunk's own text, not its questions"
    assert found.items[0].lanes.get("question") is not None and found.items[0].lanes.get("text") is None

    off = await MemoryEngine(stores, vectors, HashEmbedder(), chunk_target=4000).open()
    missed = await off.recall("s", ASKED, limit=2)
    assert all("question" not in item.lanes for item in missed.items), "with the lane off the questions are not searched"
    # Without the vector lane's guesses, the lane's rank is what puts the passage in the answer at all.
    assert [item.text for item in (await engine.recall("s", ASKED, limit=2, lanes=("text",))).items] == [HARBOUR]
    assert (await off.recall("s", ASKED, limit=2, lanes=("text",))).items == []
    assert [event.payload["question_lane"] for event in await engine.events.query("s", kind="recall")] == [True, True]
    # Fused at twice the text lane's weight, a question's first rank outranks a passage first on the words alone.
    mixed = await engine.recall("s", ASKED + " orchard apples Bramley", limit=2)
    assert mixed.items[0].lanes.get("text") is None and [item.text for item in mixed.items] == [HARBOUR, ORCHARD]


async def test_only_questions_anchored_in_the_chunk_are_kept_and_every_drop_is_counted():
    engine = await corpus(InMemoryDocumentStore())
    model = FakeChat([
        reply(("Who took the first lighthouse?", "A great fire took the lighthouse in 1850."),
              ("Where do pilots board?", "red buoy"),
              ("", QUOTE),
              (QUESTION, QUOTE),
              (QUESTION.upper(), "  The harbour at Vellmar   closes to sailing boats every November.  "),
              ("When was the lighthouse rebuilt?", "Its lighthouse was rebuilt in 1904 after a storm took the first one.")),
        "I could not think of any questions.",
    ])
    report = await engine.build_chunk_questions("s", model, per_chunk=5)
    assert (report.dropped_unquoted, report.dropped_unasked, report.dropped_repeated, report.dropped_extra) == (2, 1, 1, 1)
    assert report.dropped_unparsed == 1 and report.calls_failed == 0 and report.chunks_indexed == 1
    assert report.kept[0].questions == (QUESTION,) and report.kept[0].quotes == (QUOTE,)
    assert "Write 5 question(s)" in model.calls[0][1]
    unanchored = await engine.recall("s", "first lighthouse great fire", limit=2)
    assert all("question" not in item.lanes for item in unanchored.items), "a dropped question is never indexed"
    assert "2 whose quote" in report.text() and "1 over the per-chunk limit" in report.text()


PICKING = ("When does picking start?", "Picking starts in the last week of September and ends before the first frost.")
LIGHTHOUSE = ("When was the lighthouse rebuilt?", "Its lighthouse was rebuilt in 1904 after a storm took the first one.")


def pair(question: str, quote: str) -> str:
    return json.dumps({"question": question, "quote": quote})


async def test_a_list_the_model_never_closed_is_read_up_to_its_last_whole_question():
    from scone_memory.bench.questions import loose_pairs, parse_pairs

    # Seen from llama3.2-ctx8k: every object written, the closing bracket not.
    unclosed = "[\n  " + pair(QUESTION, QUOTE) + ",\n  " + pair(*LIGHTHOUSE) + ',\n  {"question": "Who pil'
    assert parse_pairs(unclosed) is None, "the bench's reader stays strict"
    read = loose_pairs(unclosed)
    assert read is not None and read.pairs == ((QUESTION, QUOTE), LIGHTHOUSE) and read.unread == 1
    assert loose_pairs("[" + pair(QUESTION, QUOTE)) == loose_pairs("[" + pair(QUESTION, QUOTE) + "\n")
    assert loose_pairs("See [1] first: [" + pair(QUESTION, QUOTE)).pairs == ((QUESTION, QUOTE),), \
        "a bracket in the prose is not the list"
    assert loose_pairs('[{"question": "Who pil') is None, "no whole pair, nothing read"
    assert loose_pairs("I could not think of any questions.") is None, "no list opened, nothing read"
    strings = loose_pairs("[" + pair(QUESTION, QUOTE) + ', "and a string", 7]')
    assert strings is not None and strings.pairs == ((QUESTION, QUOTE),) and strings.unread == 2, "what is not a pair is counted"
    broken = '[{"question": "When does picking start?", "quote": 7}'
    engine = await corpus(InMemoryDocumentStore())
    report = await engine.build_chunk_questions("s", FakeChat([unclosed, broken]))
    assert report.kept[0].questions == (QUESTION, LIGHTHOUSE[0])
    assert (report.read_loosely, report.dropped_unread, report.dropped_unparsed, report.chunks_indexed) == (1, 1, 1, 1)
    assert "1 read object by object" in report.text() and "1 object(s) in them that could not be read" in report.text()
    assert (report.record()["read_loosely"], report.record()["dropped_unread"]) == (1, 1)


def test_the_loose_reader_stops_at_the_lists_end_and_counts_only_objects():
    from scone_memory.bench.questions import loose_pairs

    fenced = "```json\n[" + pair(QUESTION, QUOTE) + "\n```"
    assert loose_pairs(fenced).pairs == ((QUESTION, QUOTE),) and loose_pairs(fenced).unread == 0, "a fence is not an object"
    trailing = "[" + pair(QUESTION, QUOTE) + ",\n " + pair(*LIGHTHOUSE) + ",\n]\nFor example: " + pair(*PICKING)
    read = loose_pairs(trailing)
    assert read.pairs == ((QUESTION, QUOTE), LIGHTHOUSE) and read.unread == 0, "nothing after the closing bracket is read"


async def test_a_bracket_quoted_inside_a_question_does_not_empty_the_reply():
    from scone_memory.bench.questions import loose_pairs, parse_pairs

    tags = ("What do tags look like when none were given?", "An episode stored without tags keeps tags as [] in the export.")
    export = ("How is the export laid out?", "The export writes one JSON line per episode, in id order.")
    unclosed = "[\n  " + pair(*tags) + ",\n  " + pair(*export) + ',\n  {"question": "Who'
    assert parse_pairs(unclosed) is None, "a list that does not close is not read from a bracket inside one of its strings"
    assert loose_pairs(unclosed).pairs == (tags, export)
    assert parse_pairs(pair(*tags)) is None, "an object with no list around it is not an empty list"
    assert parse_pairs("[]") == [] and parse_pairs("No questions here: [ ]") == [], "a reply of no questions still reads"
    assert parse_pairs("See [1] first: [" + pair(*export) + "]") == [export], "a bracket in the prose is not the list"
    lines = "[" + pair(*tags) + "\n" + pair(*export) + "]"
    assert parse_pairs(lines) is None and loose_pairs(lines).pairs == (tags, export) and loose_pairs(lines).unread == 0

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000,
                                question_lane=True).open()
    await engine.remember("s", "Every episode is stored with its tags. " + tags[1] + " " + export[1], kind="file", source="export.md")
    for reply_text in (unclosed, lines):
        report = await engine.build_chunk_questions("s", FakeChat([reply_text]))
        assert (report.questions, report.read_loosely, report.dropped_unparsed) == (2, 1, 0), reply_text


async def test_a_bad_object_inside_a_closed_list_loses_only_itself_and_is_counted():
    bad = '{"question": "What is the "red buoy"?", "quote": "Pilots board arriving ships two miles out, at the red buoy."}'
    closed = "[" + pair(QUESTION, QUOTE) + ", " + bad + ", " + pair(*LIGHTHOUSE) + "]"
    engine = await corpus(InMemoryDocumentStore())
    report = await engine.build_chunk_questions("s", FakeChat([closed, "[]"]))
    assert report.kept[0].questions == (QUESTION, LIGHTHOUSE[0]), "the pair after the bad object is read"
    assert (report.read_loosely, report.dropped_unread, report.dropped_unparsed) == (1, 1, 0)
    assert "never closed" not in report.text()


async def test_a_failed_call_is_counted_and_the_pass_goes_on():
    engine = await corpus(InMemoryDocumentStore())
    model = FakeChat([ChatError("timed out"), reply(PICKING)])
    report = await engine.build_chunk_questions("s", model)
    assert report.calls_failed == 1 and report.model_calls == 2
    assert report.chunks_indexed == 1 and report.kept[0].questions == (PICKING[0],)


class ByPassage:
    """A chat model that answers by what the passage it is shown holds."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return next((answer for needle, answer in self.answers.items() if needle in user), "[]")


DOC = ("# Billing rules\n\nWhat the company does with money.\n\n## Refunds\n\n"
       "A customer may ask for the amount back within thirty days. The request goes to the desk that handled "
       "the sale. Money returns the way it came, and nobody is asked why.\n\n## Late fees\n\n"
       "After the due date two percent is added.\n")
DEEP = "nobody is asked why"


JUSTIFIED = ("Must a reimbursement be justified?", "Money returns the way it came, and nobody is asked why.")


async def test_a_pass_never_touches_the_words_a_chunk_is_under(stores):
    # Ingested with the context lane on; the pass run where only the question lane is.
    ingest = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120, context_lane=True).open()
    await ingest.remember("s", DOC, source="docs/billing-rules.md")
    before = await stores.search_context("s", "refunds billing rules", 10, TextFilter())
    assert before
    asker = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120, question_lane=True).open()
    model = ByPassage({DEEP: reply(JUSTIFIED)})
    report = await asker.build_chunk_questions("s", model)
    assert report.chunks_indexed == 1 and model.calls == report.chunks_asked > 1
    assert await stores.search_context("s", "refunds billing rules", 10, TextFilter()) == before
    assert await stores.search_context("s", "reimbursement justified", 10, TextFilter()) == [], "no question in the context index"

    both = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120, context_lane=True,
                              question_lane=True).open()
    under = await both.recall("s", "refunds under the billing rules", limit=6)
    target = next(item for item in under.items if DEEP in item.text)
    assert target.lanes.get("context") is not None and target.lanes.get("text") is None, "the headings are still indexed"
    asked = await both.recall("s", "reimbursement justified", limit=3)
    assert DEEP in asked.items[0].text and asked.items[0].lanes.get("question") is not None
    assert "context" not in asked.items[0].lanes, "questions rank in their own lane"


async def test_the_question_lane_searches_questions_only_and_the_event_says_which_lanes_ran(stores):
    ingest = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120, context_lane=True).open()
    await ingest.remember("s", DOC, source="docs/billing-rules.md")
    events = InMemoryEventLog()
    asker = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120, question_lane=True,
                               events=events).open()
    unwritten = await asker.recall("s", "refunds under the billing rules", limit=6)
    assert all("context" not in item.lanes and "question" not in item.lanes for item in unwritten.items), \
        "with no pass run the question lane ranks nothing, whatever the context index holds"
    payload = (await events.query("s", kind="recall"))[-1].payload
    assert (payload["context_lane"], payload["question_lane"]) == (False, True)

    await asker.build_chunk_questions("s", ByPassage({DEEP: reply(JUSTIFIED)}))
    headings = await asker.recall("s", "refunds under the billing rules", limit=6)
    assert all("context" not in item.lanes and "question" not in item.lanes for item in headings.items)


async def test_forgetting_an_episode_while_the_model_answers_leaves_no_question_and_the_pass_says_so(stores):
    engine = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000, question_lane=True).open()
    harbour = await engine.remember("s", HARBOUR, kind="file", source="harbour.md")
    await engine.remember("s", ORCHARD, kind="file", source="orchard.md")

    class Forgets:
        async def complete(self, system: str, user: str) -> str:
            if HARBOUR in user:
                await engine.forget("s", harbour.episode_id)
                return reply((QUESTION, QUOTE))
            return reply(PICKING)

    report = await engine.build_chunk_questions("s", Forgets())
    assert (report.chunks_gone, report.chunks_indexed, report.questions) == (1, 1, 1)
    assert [kept.questions for kept in report.kept] == [(PICKING[0],)]
    assert "1 chunk(s) forgotten while the model answered" in report.text() and report.record()["chunks_gone"] == 1
    assert await stores.search_questions("s", ASKED, 5, TextFilter()) == []
    assert await stores.index_questions("s", 10_000, [QUESTION]) is False, "a chunk that is not there takes no question"
    assert await stores.index_questions("t", report.kept[0].chunk_id, [QUESTION]) is False, "nor one of another space"
    assert await stores.search_questions("t", "picking", 5, TextFilter()) == []


async def test_writing_the_lane_again_replaces_a_chunks_questions_and_turning_it_off_stops_them(stores):
    engine = await corpus(stores, context_lane=True)
    await engine.build_chunk_questions("s", FakeChat([reply((QUESTION, QUOTE)), "[]"]))
    assert [cid for cid, _ in await stores.search_questions("s", ASKED, 5, TextFilter())]
    kept = await engine.build_chunk_questions("s", FakeChat([ChatError("timed out"), "not a list"]))
    assert (kept.chunks_kept_none, kept.chunks_indexed) == (0, 0)
    assert await stores.search_questions("s", ASKED, 5, TextFilter()), "a failed or unread reply leaves what was written"

    events = InMemoryEventLog()
    off = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000, context_lane=True,
                             events=events).open()
    got = await off.recall("s", ASKED, limit=3)
    assert all("question" not in item.lanes and "context" not in item.lanes for item in got.items), \
        "with the question lane off its questions rank nothing"
    assert (await events.query("s", kind="recall"))[-1].payload["question_lane"] is False

    emptied = await engine.build_chunk_questions("s", FakeChat(["[]", reply((QUESTION, "a quote that is in no chunk at all"))]))
    assert (emptied.chunks_kept_none, emptied.chunks_indexed) == (2, 0)
    assert "2 chunk(s) kept no question and hold none now" in emptied.text() and emptied.record()["chunks_kept_none"] == 2
    assert await stores.search_questions("s", ASKED, 5, TextFilter()) == []
    held = (stores.conn.execute("SELECT count(*) FROM chunk_questions").fetchone()[0] if isinstance(stores, SqliteDocumentStore)
            else len(stores._questions["s"]))
    assert held == 0, "a chunk with no question holds no row"


async def test_the_question_lane_keeps_the_recalls_scope_and_reports_its_own_failure():
    documents = InMemoryDocumentStore()
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000, question_lane=True,
                                events=InMemoryEventLog()).open()
    await engine.remember("s", HARBOUR, kind="file", source="harbour.md", tags=["port"])
    await engine.remember("s", ORCHARD, kind="file", source="orchard.md", tags=["farm"])
    await engine.build_chunk_questions("s", FakeChat([reply((QUESTION, QUOTE)), "[]"]))
    found = await engine.recall("s", ASKED, limit=2)
    assert found.items[0].lanes.get("question") == 1
    assert "question" in (await engine.events.query("s", kind="recall"))[-1].payload["latency_ms"]
    scoped = await engine.recall("s", ASKED, limit=2, tags=["farm"])
    assert all("question" not in item.lanes for item in scoped.items), "a question outside the recall's scope ranks nothing"

    async def broken(*args, **kwargs):
        raise RuntimeError("index offline")

    documents.search_questions = broken  # type: ignore[method-assign]
    failed = await engine.recall("s", "harbour lighthouse", limit=2)
    assert failed.items and failed.items[0].text == HARBOUR
    assert "question lane: RuntimeError: index offline" in failed.degraded


async def test_forgetting_an_episode_removes_its_questions(stores):
    engine = await corpus(stores)
    await engine.build_chunk_questions("s", FakeChat([reply((QUESTION, QUOTE)), "[]"]))
    [harbour] = [item for item in (await engine.recall("s", ASKED, limit=1)).items]
    assert harbour.text == HARBOUR
    await engine.forget("s", harbour.episode_id)
    after = await engine.recall("s", ASKED, limit=2)
    assert all(item.text != HARBOUR and "context" not in item.lanes for item in after.items)
    assert await stores.search_questions("s", ASKED, 5, TextFilter()) == [], "no question outlives its chunk"
    if isinstance(stores, SqliteDocumentStore):
        assert stores.conn.execute("SELECT count(*) FROM chunk_questions").fetchone()[0] == 0
        assert stores.conn.execute("SELECT count(*) FROM chunk_questions_fts WHERE chunk_questions_fts MATCH 'mooring'").fetchone()[0] == 0
    await engine.build_chunk_questions("s", FakeChat([reply(PICKING)]))
    assert await stores.search_questions("s", "picking", 5, TextFilter())
    await stores.delete_space("s", "2026-09-15T00:00:00+00:00")
    assert await stores.search_questions("s", "picking", 5, TextFilter()) == [], "nor its space"


async def test_a_store_without_the_context_index_says_so_and_recall_still_works():
    class Plain(InMemoryDocumentStore):
        question_lane = False

    engine = await corpus(Plain())
    model = FakeChat([reply((QUESTION, QUOTE))])
    report = await engine.build_chunk_questions("s", model)
    assert not report.kept_lane and report.model_calls == 0 and model.calls == [], "no model call is spent on a lane nobody keeps"
    assert any("keeps no question index" in reason for reason in report.reasons)
    found = await engine.recall("s", "harbour lighthouse", limit=2)
    assert found.items and found.items[0].text == HARBOUR
    assert any(note.startswith("question lane:") for note in found.degraded)


async def test_the_lane_is_off_by_default_and_the_pass_refuses_while_it_is():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    assert engine.question_lane is False
    with pytest.raises(InvalidInput, match="SCONE_QUESTION_LANE"):
        await engine.build_chunk_questions("s", FakeChat())
    on = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), question_lane=True).open()
    with pytest.raises(InvalidInput, match="space name"):
        await on.build_chunk_questions("Not a space!", FakeChat())
    with pytest.raises(InvalidInput, match="question_lane"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), question_lane="yes")  # type: ignore[arg-type]


async def test_bounds_say_when_they_cut_and_a_later_pass_resumes_after_the_cut(monkeypatch):
    engine = await corpus(InMemoryDocumentStore())
    first = await engine.build_chunk_questions("s", FakeChat([reply((QUESTION, QUOTE))]), max_chunks=1)
    assert (first.chunks_total, first.chunks_asked, first.chunks_cut) == (2, 1, 1)
    assert first.resume_after == first.kept[0].chunk_id and "1 chunk(s) past max_chunks" in first.text()
    rest = await engine.build_chunk_questions("s", FakeChat(["[]"]), after_chunk=first.resume_after)
    assert (rest.chunks_total, rest.chunks_asked, rest.chunks_cut, rest.resume_after) == (1, 1, 0, None)

    monkeypatch.setattr(module, "MAX_CHUNK_BYTES", len(ORCHARD.encode()) - 1)
    skipped = await engine.build_chunk_questions("s", FakeChat(["[]"]))
    assert (skipped.skipped_long, skipped.chunks_asked, skipped.model_calls) == (1, 1, 1)
    for bad in ({"per_chunk": 0}, {"per_chunk": module.MAX_PER_CHUNK + 1}, {"per_chunk": True}, {"per_chunk": "2"},
                {"max_chunks": 0}, {"max_chunks": module.MAX_CHUNKS + 1}, {"max_chunks": True}, {"max_chunks": "2"}):
        with pytest.raises(InvalidInput):
            await engine.build_chunk_questions("s", FakeChat(), **bad)


async def test_a_pass_can_be_scoped_to_episodes():
    engine = await corpus(InMemoryDocumentStore())
    orchard = next(item for item in (await engine.recall("s", "orchard apples", limit=1)).items)
    model = FakeChat(["[]"])
    report = await engine.build_chunk_questions("s", model, episode_ids=[orchard.episode_id, orchard.episode_id])
    assert report.chunks_total == 1 and ORCHARD in model.calls[0][1], "an episode named twice is asked about once"
    assert report.model == "FakeChat", "a pass not told the model's name says what it was"


async def test_an_empty_space_asks_nothing(stores):
    engine = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), question_lane=True).open()
    report = await engine.build_chunk_questions("empty", FakeChat())
    assert (report.chunks_total, report.model_calls, report.kept, report.kept_lane) == (0, 0, (), True)


async def test_the_command_writes_the_lane_with_the_configured_model(monkeypatch):
    from scone_memory.runtime import config
    from scone_memory.runtime.config import Settings

    engine = await corpus(InMemoryDocumentStore())
    orchard = next(item for item in (await engine.recall("s", "orchard apples", limit=1)).items)
    model = FakeChat(["[]", "[]", "[]"])
    monkeypatch.setattr(config, "build_chat", lambda settings: model)
    settings = Settings.from_env({"SCONE_CHAT_MODEL": "fake-3b"})

    async def command(*options: str) -> dict:
        out = io.StringIO()
        args = build_parser().parse_args(["--json", "--space", "s", "chunk-questions", *options])
        assert await run(args, engine, io.StringIO(""), out, settings) == 0
        return json.loads(out.getvalue())

    first = await command("--per-chunk", "2", "--max-chunks", "1")
    assert (first["chunks_total"], first["chunks_cut"], first["per_chunk"], first["model"]) == (2, 1, 2, "fake-3b")
    assert "Write 2 question(s)" in model.calls[0][1]
    assert (await command("--after-chunk", str(first["resume_after"])))["chunks_total"] == 1
    assert (await command("--episode", str(orchard.episode_id)))["chunks_total"] == 1 and ORCHARD in model.calls[2][1]
    text = io.StringIO()
    monkeypatch.setattr(config, "build_chat", lambda settings: FakeChat(["[]", "[]"]))
    assert await run(build_parser().parse_args(["--space", "s", "chunk-questions"]), engine, io.StringIO(""), text, settings) == 0
    assert text.getvalue().startswith("0 question(s) kept for 0 of 2 chunk(s) asked")


async def test_the_route_and_the_command_need_a_model_and_the_lane():
    engine = await corpus(InMemoryDocumentStore())
    try:
        headers = {"Authorization": "Bearer key-a"}
        with TestClient(create_app(engine, {"key-a": "s"})) as client:
            refused = client.post("/v1/chunk-questions", headers=headers)
            assert refused.status_code == 501 and "no synthesis model" in refused.json()["error"]
        with TestClient(create_app(engine, {"key-a": "s"}, synthesis_factory=lambda: FakeChat([reply((QUESTION, QUOTE)), "[]"]))) as client:
            made = client.post("/v1/chunk-questions?per_chunk=1&max_chunks=1", headers=headers)
            assert made.status_code == 200, made.text
            body = made.json()
            assert body["chunks_indexed"] == 1 and body["kept"][0]["questions"] == [QUESTION] and body["kept_lane"] is True
            assert (body["per_chunk"], body["chunks_cut"]) == (1, 1)
            resumed = client.post(f"/v1/chunk-questions?after_chunk={body['resume_after']}", headers=headers).json()
            scoped = client.post(f"/v1/chunk-questions?episode_id={body['kept'][0]['episode_id']}", headers=headers).json()
            assert (resumed["chunks_total"], scoped["chunks_total"]) == (1, 1)
            assert client.get("/v1/recall", params={"q": ASKED, "limit": 1}, headers=headers).json()["items"][0]["text"] == HARBOUR
        with pytest.raises(InvalidInput, match="SCONE_CHAT_URL"):
            await run(build_parser().parse_args(["chunk-questions"]), engine, io.StringIO(""), io.StringIO())
    finally:
        await engine.close()
    off = await corpus(InMemoryDocumentStore(), question_lane=False)
    with TestClient(create_app(off, {"key-a": "s"}, synthesis_factory=lambda: FakeChat())) as client:
        refused = client.post("/v1/chunk-questions", headers={"Authorization": "Bearer key-a"})
        assert refused.status_code == 422 and "SCONE_QUESTION_LANE" in refused.text


async def test_a_memory_past_its_forget_after_is_not_shown_to_the_question_model():
    from scone_memory.testing import Clock

    clock = Clock("2026-09-15T12:00:00.000Z")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=4000,
                                question_lane=True, events=InMemoryEventLog(), clock=clock).open()
    await engine.remember("s", HARBOUR, kind="file", source="harbour.md", forget_after="1h")
    await engine.remember("s", ORCHARD, kind="file", source="orchard.md")
    clock.now = "2026-09-15T13:00:00.000Z"
    model = FakeChat([reply(("When does picking start?", "Picking starts in the last week of September"
                                                         " and ends before the first frost."))])
    report = await engine.build_chunk_questions("s", model)
    shown = " ".join(user for _, user in model.calls)
    assert "Vellmar" not in shown and "Bramley" in shown and report.model_calls == 1
    assert report.episodes_past_forget_after == 1 and report.record()["episodes_past_forget_after"] == 1
    assert "1 episode(s) past their forget_after not read" in report.text()
