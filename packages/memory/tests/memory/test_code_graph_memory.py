"""Remembering a source file can also record what it says about itself.

The claims are ordinary claims: cited to the line they were read from,
dated, recalled and traversable, so "what calls this?" is answered by the
graph a space already has rather than by a second tool.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"

SOURCE = '''import json


def plan(question: str) -> str:
    return tidy(question)


def tidy(text: str) -> str:
    return text.strip()
'''


async def memory(**options) -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z"), **options).open()


def triples(facts) -> set[tuple[str, str, str]]:
    return {(f.subject, f.predicate, f.object) for f in facts}


async def test_a_space_can_be_told_to_map_the_code_it_is_given():
    engine = await memory(code_graph=True)
    added = await engine.remember(SPACE, SOURCE, kind="file", source="app/planner.py")
    facts = await engine.documents.list_facts(SPACE, include_closed=True)
    assert ("app/planner.py", "defines", "app/planner.py:plan") in triples(facts)
    assert ("app/planner.py:plan", "calls", "app/planner.py:tidy") in triples(facts)
    assert all(fact.source_episode_id == added.episode_id for fact in facts)
    assert all(fact.origin == "extracted" for fact in facts), "read from the file, not stated by a person"


async def test_every_claim_is_quoted_from_the_line_it_was_read_from():
    engine = await memory(code_graph=True)
    await engine.remember(SPACE, SOURCE, kind="file", source="app/planner.py")
    [call] = [f for f in await engine.documents.list_facts(SPACE, include_closed=True)
              if f.predicate == "calls"]
    assert call.quote == "return tidy(question)"
    assert call.quote in SOURCE


async def test_a_space_that_was_not_told_to_maps_nothing():
    engine = await memory()
    await engine.remember(SPACE, SOURCE, kind="file", source="app/planner.py")
    assert await engine.documents.list_facts(SPACE, include_closed=True) == []


async def test_prose_is_not_mapped_however_it_is_configured():
    engine = await memory(code_graph=True)
    await engine.remember(SPACE, "A note about planning, which mentions def plan().", kind="note",
                          source="notes/plan.md")
    assert await engine.documents.list_facts(SPACE, include_closed=True) == []


async def test_the_graph_can_be_asked_what_calls_a_function():
    from scone_memory.entities.match import graph_match

    engine = await memory(code_graph=True)
    await engine.remember(SPACE, SOURCE, kind="file", source="app/planner.py")
    found = await graph_match(engine, SPACE, [{"subject": "?who", "predicate": "calls",
                                               "object": "app/planner.py:tidy"}])
    assert found.status == "matched", found.text
    assert [row["bindings"]["?who"]["label"] for row in found.rows] == ["app/planner.py:plan"]


async def test_remembering_the_same_file_again_does_not_double_its_claims():
    engine = await memory(code_graph=True)
    await engine.remember(SPACE, SOURCE, kind="file", source="app/planner.py")
    before = len(await engine.documents.list_facts(SPACE, include_closed=True))
    await engine.remember(SPACE, SOURCE, kind="file", source="app/planner.py")
    assert len(await engine.documents.list_facts(SPACE, include_closed=True)) == before
