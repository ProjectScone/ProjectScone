"""Entities a question resembles, found by vector, when it names none.

A question can ask about something without naming it: "which manufacturing
firm do we know?" names no entity, yet Acme Robotics is one. Each entity is
described in a line (its label, kind, strongest relations and values) and
embedded with the engine's own embedder; the question's vector finds the
nearest. It is opt-in, every such seed says it came by similarity and how
close, and the lines are embedded once each however often they are asked.
"""

from __future__ import annotations

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import similar
from scone_memory.entities.context import graph_context
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


class Counting(HashEmbedder):
    """The hash embedder, counting every text it is asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def embed(self, texts):
        self.texts.extend(texts)
        return await super().embed(texts)


async def workplace(embedder=None) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder or HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "industry", "robotics manufacturing", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "bob stone", "works_at", "Globex", valid_from=DAY)
    await engine.assert_fact("alpha", "globex", "industry", "insurance", valid_from=DAY)
    return engine


QUESTION = "which manufacturing firm do we know about?"


async def test_a_question_naming_nothing_finds_the_entity_it_resembles():
    engine = await workplace()
    unaided = await graph_context(engine, "alpha", question=QUESTION)
    aided = await graph_context(engine, "alpha", question=QUESTION, similar=True)
    assert unaided.status == "empty"
    assert aided.status == "prepared" and len(aided.seeds) >= 1
    first = next(line for line in aided.text.splitlines() if line.startswith("entity: "))
    assert first.lower().startswith("entity: acme robotics (organisation)") and " similar 0." in first
    assert aided.coverage["similar"][0]["id"] == aided.seeds[0]


async def test_a_named_entity_comes_first_and_is_never_found_twice():
    engine = await workplace()
    packet = await graph_context(engine, "alpha", question="is Globex a manufacturing firm?", similar=True)
    lines = [line for line in packet.text.splitlines() if line.startswith("entity: ")]
    assert lines[0].startswith("entity: Globex") and " similar " not in lines[0]
    assert len(packet.seeds) == len(set(packet.seeds))


async def test_a_floor_keeps_out_what_is_not_close_enough():
    engine = await workplace()
    packet = await graph_context(engine, "alpha", question=QUESTION, similar=True, min_similarity=0.99)
    assert packet.status == "empty" and packet.coverage["similar"] == []


async def test_each_line_is_embedded_once_however_often_it_is_asked(monkeypatch):
    from collections import OrderedDict

    monkeypatch.setattr(similar, "_VECTORS", OrderedDict())
    embedder = Counting()
    engine = await workplace(embedder)
    embedder.texts.clear()
    await graph_context(engine, "alpha", question=QUESTION, similar=True)
    first = len(embedder.texts)
    await graph_context(engine, "alpha", question="another question about a firm", similar=True)
    assert first > 1 and len(embedder.texts) - first == 1, "only the new question is embedded"


async def test_the_entities_left_unembedded_are_counted(monkeypatch):
    monkeypatch.setattr(similar, "MAX_SIMILAR_ENTITIES", 2)
    engine = await workplace()
    packet = await graph_context(engine, "alpha", question=QUESTION, similar=True)
    assert any(reason.startswith("similar_cut ") for reason in packet.coverage["reasons"])


def test_an_entity_is_described_by_its_kind_relations_and_values():
    from scone_memory.core.models import Fact
    from scone_memory.entities.project import project_entities

    projection = project_entities("alpha", [
        Fact(fact_id=1, space="alpha", subject="acme robotics", predicate="based_in", object="Lisbon", valid_from=DAY),
        Fact(fact_id=2, space="alpha", subject="acme robotics", predicate="industry", object="robotics manufacturing",
             valid_from=DAY),
        Fact(fact_id=3, space="alpha", subject="alice chen", predicate="works_at", object="Acme Robotics",
             valid_from=DAY)], revision=1)
    lines = similar.entity_lines(projection)
    acme = next(entity.entity_id for entity in projection.entities if entity.key == "acme robotics")
    assert lines[acme] == "acme robotics, organisation: based in Lisbon; industry robotics manufacturing; " \
                          "alice chen works at it"
