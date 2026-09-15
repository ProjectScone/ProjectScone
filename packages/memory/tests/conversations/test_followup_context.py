"""Both places that search for the last question search a follow-up with its conversation.

``integrations.chat.recall_context`` and ``realtime.context.MemoryContext``
each pick the latest user message to search. With follow-up queries on,
"since when?" after "Where does Alice Chen work?" is also searched as
"Alice Chen since when?", fused by rank with the question, inside the
same authorized scope; the receipt says what was carried and from where.
Off, nothing changes and the receipt carries no follow-up block.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.integrations.chat import recall_context
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.realtime.context import MemoryContext

GOLD = "Alice Chen has worked at Acme Robotics since May 2021."
DECOY = "Alice Chen has worked at Initech since 2015."
SINCE = ["Bob Stone has lived in Lisbon since 2019.", "Carol Diaz has led Globex since 2018.",
         "The Porto office has been open since March.", "Dan Park has coached the team since 2020.",
         "Eve Moss has kept bees since childhood.", "Frank Lee has run the night shift since June.",
         "Grace Kim has chaired the board since 2017.", "Hank Ito has owned the cafe since 2016.",
         "Iris Vale has taught piano since 2012.", "Jon Reyes has sailed the bay since 2014."]
ALICE = [{"role": "user", "content": "Where does Alice Chen work?"},
         {"role": "assistant", "content": "At Acme Robotics."}]


def turn(text: str) -> list[dict[str, object]]:
    return [*ALICE, {"role": "user", "content": text}]


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    gold = await engine.remember("alpha", GOLD, metadata={"project": "a"})
    await engine.remember("alpha", DECOY, metadata={"project": "b"})
    for text in SINCE:
        await engine.remember("alpha", text, metadata={"project": "a"})
    engine.gold_episode = gold.episode_id  # type: ignore[attr-defined]
    return engine


def spy(engine, monkeypatch) -> list[str]:
    searched: list[str] = []
    recall = engine.recall

    async def recording(space, query, **options):
        searched.append(query)
        return await recall(space, query, **options)

    monkeypatch.setattr(engine, "recall", recording)
    return searched


async def test_chat_searches_since_when_beside_the_carried_name(memory, monkeypatch):
    searched = spy(memory, monkeypatch)
    prepared, receipt = await recall_context(memory, "alpha", turn("since when?"), limit=3, where={"project": "a"},
                                             followup="carry")
    assert searched == ["since when?", "Alice Chen since when?"]
    assert receipt.followup is not None and receipt.followup["carried"] == ["Alice Chen"]
    assert receipt.followup["from_message"] == 0 and receipt.followup["query"] == "Alice Chen since when?"
    assert GOLD in prepared[0]["content"] and "Initech" not in prepared[0]["content"]


async def test_chat_off_by_default_searches_the_question_alone(memory, monkeypatch):
    searched = spy(memory, monkeypatch)
    prepared, receipt = await recall_context(memory, "alpha", turn("since when?"), limit=3, where={"project": "a"})
    assert searched == ["since when?"] and receipt.followup is None
    assert GOLD not in prepared[0]["content"]


async def test_chat_leaves_a_standalone_question_alone_and_says_so(memory, monkeypatch):
    searched = spy(memory, monkeypatch)
    _, receipt = await recall_context(memory, "alpha", turn("Which city is Globex based in?"), followup="carry")
    assert searched == ["Which city is Globex based in?"]
    assert receipt.followup is not None and receipt.followup["applied"] is False
    assert "standalone" in str(receipt.followup["reason"])


async def test_chat_reports_the_followup_even_when_nothing_is_supplied(memory):
    prepared, receipt = await recall_context(memory, "alpha", turn("since when?"), floor=1.1, followup="carry")
    assert receipt.injected is False and receipt.followup is not None and receipt.followup["carried"] == ["Alice Chen"]


async def test_chat_falls_back_from_a_failed_model_with_the_reason(memory):
    prepared, receipt = await recall_context(memory, "alpha", turn("since when?"), limit=3, where={"project": "a"},
                                             followup="rewrite",
                                             followup_model=FakeChat([ChatError("down")]))
    assert receipt.followup is not None and receipt.followup["method"] == "carry"
    assert str(receipt.followup["fallback"]).startswith("model failed: ChatError")
    assert GOLD in prepared[0]["content"]


class SlowChat:
    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(0.5)
        return json.dumps({"query": "late"})


async def test_chat_and_context_give_a_rewriting_model_its_own_deadline(memory):
    _, receipt = await recall_context(memory, "alpha", turn("since when?"), followup="rewrite",
                                      followup_model=SlowChat(), followup_timeout=0.05)
    assert receipt.followup is not None and "timeout" in str(receipt.followup["fallback"])
    context = MemoryContext(memory, "alpha", "s1", followup_queries="rewrite", followup_model=SlowChat(),
                            followup_timeout=0.05)
    _, prepared = await context.prepare(turn("since when?"))
    assert "timeout" in str(prepared["followup"]["fallback"])


async def test_chat_refuses_an_unknown_mode_or_a_rewrite_without_a_model_before_any_question(memory):
    nothing_asked = [{"role": "system", "content": "Be terse."}]
    with pytest.raises(InvalidInput):
        await recall_context(memory, "alpha", nothing_asked, followup="sometimes")
    with pytest.raises(InvalidInput):
        await recall_context(memory, "alpha", nothing_asked, followup="rewrite")


async def test_context_carries_within_its_scope_and_says_what_it_carried(memory, monkeypatch):
    searched = spy(memory, monkeypatch)
    context = MemoryContext(memory, "alpha", "s1", where={"project": "a"}, limit=2, followup_queries="carry")
    _, receipt = await context.prepare(turn("since when?"))
    assert searched == ["since when?", "Alice Chen since when?"]
    assert receipt["followup"]["carried"] == ["Alice Chen"] and receipt["followup"]["from_message"] == 0
    referenced = {reference["episode_id"] for reference in receipt["references"]}
    assert memory.gold_episode in referenced
    texts = json.dumps(receipt["evidence_graph"])
    assert "Initech" not in texts


async def test_context_off_searches_the_question_alone_with_no_followup_block(memory, monkeypatch):
    searched = spy(memory, monkeypatch)
    _, receipt = await MemoryContext(memory, "alpha", "s1", where={"project": "a"}, limit=2).prepare(turn("since when?"))
    assert searched == ["since when?"] and "followup" not in receipt
    assert memory.gold_episode not in {reference["episode_id"] for reference in receipt["references"]}


async def test_context_leaves_a_standalone_question_alone(memory, monkeypatch):
    searched = spy(memory, monkeypatch)
    _, receipt = await MemoryContext(memory, "alpha", "s1", followup_queries="carry").prepare(
        turn("Which city is Globex based in?"))
    assert searched == ["Which city is Globex based in?"] and receipt["followup"]["applied"] is False


async def test_context_falls_back_from_a_failed_model_and_still_prepares(memory):
    context = MemoryContext(memory, "alpha", "s1", limit=2, followup_queries="rewrite",
                            followup_model=FakeChat([ChatError("down")]))
    _, receipt = await context.prepare(turn("since when?"))
    assert receipt["status"] == "prepared" and receipt["followup"]["method"] == "carry"
    assert str(receipt["followup"]["fallback"]).startswith("model failed: ChatError")


async def test_context_searches_a_follow_up_that_alone_would_read_as_an_overview(memory):
    await memory.remember("alpha", "Project Atlas is waiting on the security review.")
    messages = [{"role": "user", "content": "Tell me about Project Atlas."},
                {"role": "assistant", "content": "It is a migration."},
                {"role": "user", "content": "What's the status of it?"}]
    _, plain = await MemoryContext(memory, "alpha", "s1").prepare(messages)
    assert plain["retrieval_mode"] == "overview"
    _, carried = await MemoryContext(memory, "alpha", "s1", followup_queries="carry").prepare(messages)
    assert carried["retrieval_mode"] == "search" and carried["followup"]["carried"] == ["Project Atlas"]
    assert "overview" in carried["followup"]["reason"]


async def test_context_does_not_search_a_follow_up_query_that_also_reads_as_an_overview(memory, monkeypatch):
    messages = [{"role": "user", "content": "Tell me about Project."},
                {"role": "user", "content": "What's the status of it?"}]
    searched = spy(memory, monkeypatch)
    _, receipt = await MemoryContext(memory, "alpha", "s1", followup_queries="carry").prepare(messages)
    assert receipt["retrieval_mode"] == "overview" and searched == []
    assert receipt["followup"]["applied"] is False and receipt["followup"]["query"] is None
    assert "also reads as an overview" in receipt["followup"]["reason"]


@pytest.mark.parametrize("options", [{"followup_queries": "sometimes"}, {"followup_queries": "rewrite"},
    {"followup_queries": "carry", "followup_timeout": True}, {"followup_queries": "carry", "followup_timeout": "5"},
    {"followup_model": FakeChat()}, {"followup_queries": "rewrite", "followup_model": FakeChat(), "followup_timeout": 0},
    {"followup_queries": "carry", "followup_timeout": float("nan")},
    {"followup_queries": "rewrite", "followup_model": FakeChat(), "followup_timeout": 61}])
def test_invalid_followup_settings_rejected(memory, options):
    with pytest.raises(ValueError):
        MemoryContext(memory, "alpha", "one", **options)


def test_adaptive_retrieval_plans_its_own_queries(memory):
    from scone_memory.retrieval.adaptive import AdaptiveRetriever

    with pytest.raises(ValueError, match="adaptive"):
        MemoryContext(memory, "alpha", "one", adaptive_retriever=AdaptiveRetriever(memory, object()),
                      recall_timeout=30, followup_queries="carry")


def test_text_conversation_passes_the_setting_to_its_context_and_refuses_tool_mode(memory):
    from scone_memory.realtime.text import TextConversation

    conversation = TextConversation(memory, "alpha", "s1", lambda: None, followup_queries="carry")
    assert conversation._context._followup == "carry"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="follow"):
        TextConversation(memory, "alpha", "s1", tool_model_factory=lambda: None, followup_queries="carry")
