"""Entities a question resembles, found by vector, when it names none.

A question can ask about something without naming it: "which manufacturing
firm do we know?" names no entity, yet Acme Robotics is one. Each entity is
described in a line (its label, kind, strongest relations and values) and
embedded with the engine's own embedder; the question's vector finds the
nearest. It is opt-in, every such seed says it came by similarity and how
close, and the lines are embedded once each however often they are asked.
"""

from __future__ import annotations

import pytest

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


class Faulty(HashEmbedder):
    """The hash embedder until armed; then an answer that cannot be used."""

    def __init__(self, fault: str, dim: int = 256) -> None:
        super().__init__(dim)
        self.fault, self.armed = fault, False

    async def embed(self, texts):
        vectors = await super().embed(texts)
        if not self.armed:
            return vectors
        if self.fault == "raises":
            raise RuntimeError("the provider is down")
        if self.fault == "short":
            return vectors[:-1]
        if self.fault == "nan":
            return [[float("nan"), *vector[1:]] for vector in vectors]
        if self.fault == "infinite":
            return [[float("inf"), *vector[1:]] for vector in vectors]
        if self.fault == "narrow":
            return [vector[:1] for vector in vectors]
        return [[True, *vector[1:]] for vector in vectors]


@pytest.mark.parametrize("fault, said", [
    ("raises", "(RuntimeError)"), ("short", "answered 5 texts with 4 vectors"), ("nan", "not a finite number"),
    ("infinite", "not a finite number"), ("narrow", "of 1 values, not 256"), ("not_a_number", "not a finite number")])
async def test_an_unusable_embedder_answer_is_said_and_never_kept(monkeypatch, fault, said):
    """Named seeds stay; the similar ones are said to be unavailable, never
    silently absent, and nothing the embedder said is kept for later."""
    from collections import OrderedDict

    monkeypatch.setattr(similar, "_VECTORS", OrderedDict())
    embedder = Faulty(fault)
    engine = await workplace(embedder)
    embedder.armed = True
    packet = await graph_context(engine, "alpha", question="is Globex a manufacturing firm?", similar=True)
    assert packet.status == "prepared" and packet.coverage["similar"] == []
    [entity] = [line for line in packet.text.splitlines() if line.startswith("entity: ")]
    assert entity.startswith("entity: Globex")
    [reason] = [reason for reason in packet.coverage["reasons"] if reason.startswith("similar_unavailable")]
    assert said in reason and "provider is down" not in packet.text
    assert "coverage: limited: " in packet.text and reason in packet.text
    assert similar._VECTORS == OrderedDict()


async def test_an_unusable_answer_is_refused_before_any_of_it_is_scored():
    embedder = Faulty("narrow", dim=2)
    embedder.armed = True
    engine = type("Engine", (), {"embedder": embedder})()
    with pytest.raises(similar.SimilarUnavailable, match="1 values, not 2"):
        await similar._vectors(engine, ["first text", "second text"])


class Sized:
    """One model name at two sizes: vectors of one size never answer for
    the other."""

    id = "shared-model-name"

    def __init__(self, dim: int) -> None:
        self.dim, self.texts = dim, []

    async def embed(self, texts):
        self.texts.extend(texts)
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


async def test_one_model_at_two_sizes_is_embedded_at_each(monkeypatch):
    from collections import OrderedDict

    monkeypatch.setattr(similar, "_VECTORS", OrderedDict())
    narrow, wide = Sized(2), Sized(3)
    first = await similar._vectors(type("Engine", (), {"embedder": narrow})(), ["shared"])
    second = await similar._vectors(type("Engine", (), {"embedder": wide})(), ["shared"])
    assert [len(first[0]), len(second[0])] == [2, 3] and wide.texts == ["shared"]


async def test_a_vector_evicted_while_a_request_waits_is_still_its_answer(monkeypatch):
    """A request that found a vector kept holds it across its own wait for
    the embedder, so another request filling the cache meanwhile cannot
    take it away."""
    import asyncio
    from collections import OrderedDict

    monkeypatch.setattr(similar, "_VECTORS", OrderedDict())
    monkeypatch.setattr(similar, "_KEPT", 2)
    embedder = HashEmbedder(dim=8)
    engine = type("Engine", (), {"embedder": embedder})()
    [kept] = await similar._vectors(engine, ["shared line"])
    started, release = asyncio.Event(), asyncio.Event()

    class Waiting(HashEmbedder):
        async def embed(self, texts):
            started.set()
            await release.wait()
            return await super().embed(texts)

    waiting = type("Engine", (), {"embedder": Waiting(dim=8)})()
    pending = asyncio.create_task(similar._vectors(waiting, ["shared line", "a new question"]))
    await started.wait()
    await similar._vectors(engine, ["another line", "and one more"])
    release.set()
    shared, _ = await pending
    assert shared == kept


async def test_numbers_of_any_real_type_are_numbers(monkeypatch):
    """An embedder built on numpy answers in numpy floats; they are numbers."""
    from collections import OrderedDict
    import numpy

    monkeypatch.setattr(similar, "_VECTORS", OrderedDict())

    class Numpy(HashEmbedder):
        async def embed(self, texts):
            return [[numpy.float32(value) for value in vector] for vector in await super().embed(texts)]

    [vector] = await similar._vectors(type("Engine", (), {"embedder": Numpy(dim=4)})(), ["acme robotics"])
    assert len(vector) == 4 and all(type(value) is float for value in vector)
