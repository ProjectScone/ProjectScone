"""The vector lane's voice in rank fusion is a setting on the record, not a fact of nature.

Reciprocal rank fusion gives every lane the same voice. With an embedder
whose vectors are hashed tokens, the vector lane is a weak echo of the
text lane, and its confident wrong picks can outvote the text lane's
right ones. ``vector_weight`` lets a caller say how much that lane
counts, and every recall's event says what it was. Unset, it follows the
embedder: a hundredth voice for hashed tokens, measured on LongMemEval-S
(benchmarks/northstar-defaults-2026-09-14.results.md), a full voice for
any other embedder.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput

BILLING = "The billing run went out late and the invoices were wrong."


async def engine_with(**kw):
    events = InMemoryEventLog()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=events, **kw).open()
    await engine.remember("s", BILLING)
    await engine.remember("s", "Invoices invoices invoices: the word again and again, about nothing.")
    await engine.remember("s", "A shed roof before winter.")
    return engine


async def test_a_lighter_vector_lane_lets_the_text_lane_decide_when_they_disagree():
    heavy = await engine_with(vector_weight=1.0)
    light = await engine_with(vector_weight=0.25)
    query = "billing run invoices"
    default = await heavy.recall("s", query, limit=3)
    weighted = await light.recall("s", query, limit=3)
    assert [i.lanes for i in default.items] == [i.lanes for i in weighted.items], "the lanes rank the same; only the fusion changed"
    text_first = min(weighted.items, key=lambda i: i.lanes.get("text", 10**6))
    assert weighted.items[0].chunk_id == text_first.chunk_id, "the text lane's first choice wins at a quarter weight"


async def test_the_weight_is_on_every_recall_event_and_is_bounded():
    engine = await engine_with(vector_weight=0.5)
    await engine.recall("s", "billing", limit=1)
    [event] = await engine.events.query("s", kind="recall")
    assert event.payload["fusion_weights"] == {"vector": 0.5, "text": 1.0}
    plain = await engine_with()
    await plain.recall("s", "billing", limit=1)
    [event] = await plain.events.query("s", kind="recall")
    assert event.payload["fusion_weights"] == {"vector": 0.01, "text": 1.0}, "unset, a hashed-token embedder's lane gets a hundredth voice"
    for bad in (0, -1, 4.5, "1", True):
        with pytest.raises(InvalidInput, match="vector_weight"):
            MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), vector_weight=bad)  # type: ignore[arg-type]


async def test_the_default_voice_follows_the_embedder_and_is_on_the_record():
    from scone_memory.memory.engine import HASHED_VECTOR_WEIGHT

    hashed = await engine_with()
    assert hashed.vector_weight == HASHED_VECTOR_WEIGHT == 0.01, "a hashed-token embedder is a weak echo of the text lane"
    await hashed.recall("s", "billing", limit=1)
    [event] = await hashed.events.query("s", kind="recall")
    assert event.payload["fusion_weights"] == {"vector": 0.01, "text": 1.0}

    class Real:
        """An embedder that is not a hash of tokens, as far as the engine can tell: its id says so."""
        id = "real-embedder-v1"
        dim = HashEmbedder().dim

        async def embed(self, texts):
            return await HashEmbedder().embed(texts)

    real = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Real())  # type: ignore[arg-type]
    assert real.vector_weight == 1.0, "an embedder that knows things keeps its full voice"
    said = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), vector_weight=1.0)
    assert said.vector_weight == 1.0, "a caller who says a weight is heard over the rule"


#: A passage that answers in many words: second for the text lane, far down
#: for hashed vectors, whose cosine favours short passages that repeat a
#: query word.
ANSWER = ("Last spring the billing run for the northern warehouses slipped by a week, so every customer "
          "received late invoices, and the finance team spent the whole of April reconciling accounts, "
          "answering complaints from suppliers, and rewriting the payment schedule for the quarter.")
ECHOES = ("The run of invoices.", "Invoices pile up on the desk.", "The run was long.", "Invoices and receipts.",
          "A run in the park.", "Invoices are paper.", "Morning run, cold air.", "Invoices, again.")


async def test_at_the_hashed_default_the_vector_lane_no_longer_overturns_the_text_lanes_order():
    async def text_ranks(**kw) -> list[int]:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), **kw).open()
        for text in (ANSWER, *ECHOES):
            await engine.remember("s", text)
        result = await engine.recall("s", "billing run invoices", limit=3)
        return [item.lanes["text"] for item in result.items]

    assert await text_ranks(vector_weight=0.25) != [1, 2, 3], (
        "the fixture has the junction: at a quarter voice a short echo the vector lane ranks far above the answer "
        "takes the place the text lane gave the answer")
    assert await text_ranks() == [1, 2, 3], "unset, the text lane's order stands"


#: The clock, and two passages: the one the query names exactly is forty
#: minutes older than one that shares no word with it. At a hundredth
#: voice the vector lane's rank scores are smaller than the recency term's
#: difference over those forty minutes.
NOW = "2026-09-15T12:00:00Z"
EXACT = "quarterly invoice reconciliation northern warehouses"
UNRELATED = "lunch weather notes today"


async def _answers_nothing(*args, **kwargs):
    return []


async def _fails(*args, **kwargs):
    raise RuntimeError("text index down")


@pytest.mark.parametrize("text_lane", ["not-asked", "failed", "found-nothing"], ids=["not-asked", "failed", "found-nothing"])
async def test_with_nothing_from_the_text_lane_the_vector_lane_speaks_at_full_voice(text_lane):
    from scone_memory.memory.engine import Record

    async def recall(**kw):
        events = InMemoryEventLog()
        documents = InMemoryDocumentStore()
        engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), events=events,
                                    clock=lambda: NOW, **kw).open()
        await engine.remember_many("s", [Record(content=EXACT, created_at="2026-09-15T11:20:00Z"),
                                         Record(content=UNRELATED, created_at="2026-09-15T12:00:00Z")])
        lanes: tuple[str, ...] = ("vector", "text")
        if text_lane == "not-asked":
            lanes = ("vector",)
        else:
            stand_in = _fails if text_lane == "failed" else _answers_nothing
            documents.search_text = stand_in  # type: ignore[method-assign]
            documents.search_terms = stand_in  # type: ignore[attr-defined]
        result = await engine.recall("s", EXACT, limit=2, lanes=lanes)
        [event] = await events.query("s", kind="recall")
        return [item.text for item in result.items], event.payload["fusion_weights"], result.degraded

    heard, weights, degraded = await recall(vector_weight=1.0)
    assert heard == [EXACT, UNRELATED] and weights == {"vector": 1.0, "text": 1.0}
    assert ("text: RuntimeError: text index down" in degraded) == (text_lane == "failed"), "the fixture takes the path it names"
    ordered, weights, _ = await recall()
    assert ordered == [EXACT, UNRELATED], (
        "a voice against the text lane has nothing to speak against: the vector lane's order stands, and recency "
        "stays a tie-breaker, not the order")
    assert weights == {"vector": 1.0, "text": 1.0}, "the event records the voice the lane had"


async def test_a_text_lane_that_brings_passages_keeps_the_hundredth_voice_on_the_record():
    engine = await engine_with()
    await engine.recall("s", "billing", limit=3, lanes=("vector", "text"))
    [event] = await engine.events.query("s", kind="recall")
    assert event.payload["fusion_weights"] == {"vector": 0.01, "text": 1.0}
