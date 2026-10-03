"""The second hop where callers meet it: the answer route and the agent's search tool.

Both are opt-in. A question that compares things it names is searched once (``second_hop.should_hop``), and the
tool's second search runs under the same scope as its first, so a hop cannot reach memory the caller excluded.
"""
from __future__ import annotations

import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope
from scone_memory.retrieval.router import answer_question
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"
BRIDGE = "Where was the director of Inception born?"
COMPARISON = "Is Inception or Memento the older film?"


async def memory() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                               clock=Clock("2024-07-01T00:00:00.000Z")).open()
    for text in ("Inception is a 2010 film directed by Christopher Nolan.",
                 "Christopher Nolan was born in Westminster, London.",
                 "Memento is a 2000 film about memory.",
                 "The harbour crane was repainted in May."):
        await engine.remember(SPACE, text, metadata={"team": "blue"})
    await engine.remember(SPACE, "Christopher Nolan PRIVATE_MARKER salary notes.", metadata={"team": "red"})
    return engine


async def test_the_answer_route_hops_only_when_asked_and_reports_it() -> None:
    engine = await memory()
    plain = await answer_question(engine, SPACE, BRIDGE, route="recall")
    assert "hop" not in plain.detail
    hopped = await answer_question(engine, SPACE, BRIDGE, route="recall", hop=True)
    hop = hopped.detail["hop"]
    assert isinstance(hop, dict) and hop["ran"] is True and "seed_chunk_id" in hop
    compared = await answer_question(engine, SPACE, COMPARISON, route="recall", hop=True)
    assert compared.detail["hop"] == {"ran": False}


async def test_the_search_tool_hops_within_the_callers_scope() -> None:
    engine = await memory()
    scope = RecallScope.validated(where={"team": "blue"})
    plain = await ScopedMemoryTools(engine, SPACE, scope=scope, timeout_s=30).run("search_memory", {"query": BRIDGE})
    assert "hop" not in plain
    tools = ScopedMemoryTools(engine, SPACE, scope=scope, timeout_s=30, second_hop=True)
    hopped = await tools.run("search_memory", {"query": BRIDGE})
    assert hopped["ok"] is True and isinstance(hopped["hop"], dict) and hopped["hop"]["ran"] is True
    # The hop's query names Christopher Nolan, as the excluded passage does; the scope still holds.
    assert "PRIVATE_MARKER" not in json.dumps(hopped)
    compared = await tools.run("search_memory", {"query": COMPARISON})
    assert compared["hop"] == {"ran": False, "reason": "comparison"}


async def test_second_hop_must_be_a_boolean() -> None:
    with pytest.raises(ValueError):
        ScopedMemoryTools(await memory(), SPACE, scope=RecallScope.validated(), second_hop="yes")  # type: ignore[arg-type]
