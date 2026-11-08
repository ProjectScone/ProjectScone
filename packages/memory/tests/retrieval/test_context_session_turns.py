"""The context shows the turns around a recalled turn, from its own session.

The search is fixed to the turns each test names, so what is tested is what
the context does with them: the turns either side are read from the store,
verified like any passage, shown right after their turn beyond ``limit``,
and counted in the receipt, including those a bound kept out.
"""

import asyncio
import json

import pytest

from scone_memory import Record
from scone_memory.core.models import RecallItem, RecallResult
from scone_memory.realtime.context import MemoryContext
from scone_memory.retrieval.evidence_graph import MAX_CHUNKS

LINES = ["Morning! Did you sleep well after the trip?",
         "Not really, the audit kept me up most of the night.",
         "Oh no. What happened with it?",
         "We had to move the harbour launch to the fourteenth of March.",
         "Because the auditors found two open tickets.",
         "That sounds stressful, take it easy today."]
ASK = [{"role": "user", "content": "When is the harbour launch?"}]


async def said(engine, session, lines, extra=None):
    """The turns of ``session`` written in order, ``extra`` metadata by
    turn number; their episode ids."""
    written = await engine.remember_many("alpha", [
        Record(content=text, kind="conversation", source=f"{session}:{n}",
               metadata={"session_id": session, "turn_id": f"{session}:{n}", **(extra or {}).get(n, {})})
        for n, text in enumerate(lines)])
    return [result.episode_id for result in written]


async def item(engine, episode_id):
    episode = await engine.documents.get_episode("alpha", episode_id)
    chunk = (await engine.documents.chunks_of("alpha", episode_id))[0]
    return RecallItem(chunk_id=chunk.chunk_id, episode_id=episode_id, text=chunk.text, score=1.0,
                      source=episode.source, created_at=chunk.created_at, metadata=dict(episode.metadata))


async def search_finds(engine, monkeypatch, episode_ids):
    found = [await item(engine, episode_id) for episode_id in episode_ids]

    async def recall(*args, **kwargs):
        return RecallResult(items=found)

    monkeypatch.setattr(engine, "recall", recall)
    return found


async def shown(engine, receipt):
    return [(await engine.documents.get_episode("alpha", reference["episode_id"])).source
            for reference in receipt["references"]]


async def test_a_recalled_turn_is_shown_with_the_turns_either_side_after_it(engine, monkeypatch):
    turns = await said(engine, "a", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])
    _, before = await MemoryContext(engine, "alpha", "current", limit=1).prepare(ASK)
    assert await shown(engine, before) == ["a:3"] and "session_turns" not in before
    request, after = await MemoryContext(engine, "alpha", "current", limit=1, session_turns=1).prepare(ASK)
    assert await shown(engine, after) == ["a:3", "a:2", "a:4"]
    payload = json.loads(request[0]["content"].split("\n", 1)[1])
    assert [source["text"] for source in payload["sources"]] == [LINES[3], LINES[2], LINES[4]]
    assert after["session_turns"] == {"seeds": 1, "found": 2, "added": 2, "moved": 0, "not_looked": 0,
                                      "no_session": 0, "short": 0, "shortened": 0, "skipped": None,
                                      "cut": 0, "shown": 2}
    assert after["omitted_count"] == 0


async def test_brought_turns_never_crowd_out_a_ranked_turn(engine, monkeypatch):
    first = await said(engine, "a", LINES)
    second = await said(engine, "b", ["Hey, back from the coast now.", "Bought a kite for my nephew."])
    await search_finds(engine, monkeypatch, [first[3], second[0]])
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=2, session_turns=1).prepare(ASK)
    assert await shown(engine, receipt) == ["a:3", "a:2", "a:4", "b:0", "b:1"]


async def test_brought_turns_take_only_the_bytes_the_ranked_turns_leave(engine, monkeypatch):
    long_reply = ("We went over every line of the audit together, page by page, and in the end found "
                  "nothing alarming in it at all, which was a relief to everyone in the room.")
    first = await said(engine, "a", [LINES[2], LINES[3], long_reply])
    second = await said(engine, "b", ["Back now.", "A kite."])
    await search_finds(engine, monkeypatch, [first[1], second[0]])
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=2, max_context_bytes=1000).prepare(ASK)
    assert await shown(engine, receipt) == ["a:1", "b:0"]
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=2, max_context_bytes=1000,
                                     session_turns=1).prepare(ASK)
    # Shown in order, a:1's turns before b:0, all four fit; a:2 does not
    # fit beside both ranked turns and would have taken b:0's bytes.
    assert await shown(engine, receipt) == ["a:1", "a:0", "b:0", "b:1"]


async def test_turns_past_the_verified_passage_cap_are_cut_and_counted(engine, monkeypatch):
    turns = await said(engine, "a", [f"Line {n} of a long chat about the harbour." for n in range(50)])
    ranked = turns[2::5]
    assert len(ranked) == 10
    await search_finds(engine, monkeypatch, ranked)
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=10, max_context_bytes=64000,
                                     session_turns=2).prepare(ASK)
    sources = await shown(engine, receipt)
    # Each ranked turn brings four of its own; fourteen fit under the cap,
    # the first ranked turns' first, and every ranked turn is shown.
    assert len(sources) == MAX_CHUNKS == 10 + 14
    assert sources[:18] == ["a:2", "a:0", "a:1", "a:3", "a:4", "a:7", "a:5", "a:6", "a:8", "a:9",
                            "a:12", "a:10", "a:11", "a:13", "a:14", "a:17", "a:15", "a:16"]
    assert sources[18:] == [f"a:{n}" for n in range(22, 50, 5)]
    assert (receipt["session_turns"]["found"], receipt["session_turns"]["cut"],
            receipt["session_turns"]["shown"]) == (40, 26, 14)
    assert receipt["omitted_count"] == 0


async def test_turns_of_the_current_session_still_in_the_window_are_not_shown(engine, monkeypatch):
    turns = await said(engine, "current", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])
    context = MemoryContext(engine, "alpha", "current", limit=1, session_turns=1)
    _, receipt = await context.prepare(ASK, admit_turn_ids=frozenset({"current:3"}))
    assert await shown(engine, receipt) == ["current:3"]
    _, receipt = await context.prepare(ASK, admit_turn_ids=frozenset({"current:2", "current:3"}))
    assert await shown(engine, receipt) == ["current:3", "current:2"]


async def test_a_turn_outside_the_contexts_scope_is_not_shown(engine, monkeypatch):
    owned = await said(engine, "c", LINES, {n: {"user": "bob" if n in (2, 4) else "ann"} for n in range(6)})
    await search_finds(engine, monkeypatch, [owned[3]])
    _, receipt = await MemoryContext(engine, "alpha", "current", where={"user": "ann"}, limit=1,
                                     session_turns=2).prepare(ASK)
    assert await shown(engine, receipt) == ["c:3", "c:1", "c:5"]
    assert receipt["session_turns"]["found"] == 2, "bob's turns are not read as found"


async def test_turns_the_context_may_not_show_take_no_room_and_are_not_cut(engine, monkeypatch):
    turns = await said(engine, "current", [f"Line {n} of a long chat about the harbour." for n in range(50)])
    await search_finds(engine, monkeypatch, turns[2::5])
    admitted = frozenset(f"current:{n}" for n in range(2, 50, 5))
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=10, max_context_bytes=64000,
                                     session_turns=2).prepare(ASK, admit_turn_ids=admitted)
    assert len(receipt["references"]) == 10
    assert (receipt["session_turns"]["cut"], receipt["session_turns"]["shown"]) == (0, 0)


async def test_a_turn_dated_after_the_engines_now_is_not_shown(engine, monkeypatch):
    written = await engine.remember_many("alpha", [
        Record(content=text, kind="conversation", source=f"a:{n}", metadata={"session_id": "a"},
               created_at=f"2025-01-01T10:0{n}:00Z") for n, text in enumerate(LINES)])
    await search_finds(engine, monkeypatch, [written[3].episode_id])
    engine.clock.now = "2025-01-01T10:03:30Z"
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=1, session_turns=1).prepare(ASK)
    assert await shown(engine, receipt) == ["a:3", "a:2"]


async def test_a_failing_store_keeps_the_ranked_turn_and_says_so(engine, monkeypatch):
    turns = await said(engine, "a", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])

    async def unavailable(*args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(engine.documents, "page_episodes", unavailable)
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=1, session_turns=1).prepare(ASK)
    assert await shown(engine, receipt) == ["a:3"]
    assert receipt["session_turns"]["skipped"] == "the store failed: RuntimeError"
    assert receipt["session_turns"]["shown"] == 0
    assert receipt["degraded"] == []


async def test_a_slow_store_times_out_and_keeps_the_ranked_turn(engine, monkeypatch):
    turns = await said(engine, "a", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])

    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(engine.documents, "page_episodes", slow)
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=1, session_turns=1,
                                     recall_timeout=0.2).prepare(ASK)
    assert await shown(engine, receipt) == ["a:3"]
    assert receipt["session_turns"]["skipped"] == "timeout"


async def test_a_failed_evidence_check_drops_the_brought_turns(engine, monkeypatch):
    import scone_memory.realtime.context as module
    turns = await said(engine, "a", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])

    async def unavailable(*args, **kwargs):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(module, "build_query_evidence_graph", unavailable)
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=1, session_turns=1).prepare(ASK)
    assert await shown(engine, receipt) == ["a:3"]
    assert receipt["session_turns"]["shown"] == 0
    assert "passage_window_status" not in receipt, "chunk windows were off"


async def test_session_turns_and_chunk_windows_combine(engine, monkeypatch):
    turns = await said(engine, "a", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])
    _, receipt = await MemoryContext(engine, "alpha", "current", limit=1, session_turns=1,
                                     neighbor_chunks=1).prepare(ASK)
    assert await shown(engine, receipt) == ["a:3", "a:2", "a:4"]


@pytest.mark.parametrize("radius", [-1, 5, True, 1.5, "1"], ids=repr)
async def test_session_turns_is_bounded(engine, radius):
    with pytest.raises(ValueError, match="session_turns"):
        MemoryContext(engine, "alpha", "current", session_turns=radius)


async def test_session_turns_refuses_an_order_that_would_separate_them(engine):
    with pytest.raises(ValueError, match="reading_order"):
        MemoryContext(engine, "alpha", "current", session_turns=1, reading_order="ends")


async def test_session_turns_refuses_adaptive_retrieval(engine):
    from ..retrieval.test_adaptive_context import ScriptedAssessor, retriever, select_all
    with pytest.raises(ValueError, match="adaptive"):
        MemoryContext(engine, "alpha", "current", session_turns=1,
                      adaptive_retriever=retriever(engine, ScriptedAssessor(select_all)))


async def test_a_text_conversation_gives_the_model_the_turns_either_side(engine, monkeypatch):
    from scone_memory.realtime.text import TextConversation
    from ..conversations.test_text_conversation import ScriptedModel
    turns = await said(engine, "a", LINES)
    await search_finds(engine, monkeypatch, [turns[3]])
    model = ScriptedModel()
    conversation = TextConversation(engine, "alpha", "current", lambda: model, session_turns=1)
    result = await conversation.reply("When is the harbour launch?")
    assert result["memory_context"]["session_turns"]["shown"] == 2
    payload = next(message["content"] for message in model.requests[0] if "Scone retrieved" in message["content"])
    assert LINES[4] in payload


async def test_tool_mode_refuses_session_turns(engine):
    from scone_memory.realtime.text import TextConversation
    from ..conversations.test_text_tool_answer import Script
    with pytest.raises(ValueError, match="session_turns"):
        TextConversation(engine, "alpha", "current", tool_model_factory=lambda: Script(), session_turns=1)
