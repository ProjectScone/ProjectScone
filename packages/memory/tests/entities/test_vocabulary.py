"""Which vocabulary a graph was built under, and where it came from.

What a predicate means to another predicate decides which edges a reader
is shown, and that configuration is per-process: two processes reading
one space can hand two readers different graphs. This does not fix that.
What it pins is that the disagreement is **discoverable** -- every answer
says which vocabulary it used and where that came from -- and that a view
cached under one vocabulary is never served for another.

Storing a vocabulary in the space is the actual fix and is not here. The
module docstring records why the event log cannot serve as its home:
retention is a property of the store, not of the handle that opened it,
so no honest promise to keep a record is available to make.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.meanings import RelationMeanings
from scone_memory.entities.vocabulary import read_vocabulary

pytestmark = pytest.mark.asyncio

EMPLOYS = RelationMeanings(inverse={"works_at": "employs"})


async def memory(meanings=None):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              relation_meanings=meanings).open()


async def test_a_configured_process_says_the_vocabulary_is_its_own():
    engine = await memory(meanings=EMPLOYS)
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "process" and held.meanings is not None
        assert held.meanings.opposite("works_at") == "employs"
        assert "another process configured differently" in held.why, held.why
    finally:
        await engine.close()


async def test_an_unconfigured_process_says_there_are_none():
    engine = await memory()
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "none" and held.meanings is None
        assert "only what was said" in held.why, held.why
    finally:
        await engine.close()


async def test_reading_the_vocabulary_touches_no_store():
    """It is this process's configuration, so there is nothing to read. A
    store query per graph request would be cost with nothing bought."""
    engine = await memory(meanings=EMPLOYS)
    reads = []
    original = engine.documents.revision

    async def counted(space):
        reads.append(space)
        return await original(space)

    engine.documents.revision = counted
    try:
        await read_vocabulary(engine, "default")
        assert reads == [], reads
    finally:
        engine.documents.revision = original
        await engine.close()


async def test_an_explicitly_empty_vocabulary_is_not_the_absence_of_one():
    """`RelationMeanings()` is falsy, so testing it for truth reports "no
    vocabulary" for a vocabulary that says there are no meanings."""
    engine = await memory(meanings=RelationMeanings())
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "process", held.why
        said = held.record()
        assert isinstance(said["meanings"], dict) and said["meanings"]["inverse"] == {}, said
    finally:
        await engine.close()


async def test_two_different_vocabularies_have_two_identities():
    """The identity is what keeps a view built under one set of meanings
    from being served for another."""
    one = await memory(meanings=EMPLOYS)
    two = await memory(meanings=RelationMeanings(symmetric=["married_to"]))
    none = await memory()
    try:
        seen = {(await read_vocabulary(engine, "default")).identity for engine in (one, two, none)}
        assert len(seen) == 3, seen
    finally:
        for engine in (one, two, none):
            await engine.close()


async def test_the_projection_is_built_under_the_vocabulary_in_force():
    """The wiring that makes the identity worth having: the graph a reader
    gets reflects the meanings in force, and says so."""
    from scone_memory.entities.read import load_projection

    engine = await memory(meanings=EMPLOYS)
    try:
        said = await engine.remember("default", "Ana Alves works at Meridian Health.")
        await engine.assert_fact("default", "ana alves", "works_at", "Meridian Health",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=said.episode_id,
                                 quote="Ana Alves works at Meridian Health.")
        found, _ = await load_projection(engine, "default", mode="current")
        assert any(one.predicate == "employs" for one in found.implied), \
            [one.predicate for one in found.implied]
    finally:
        await engine.close()


async def test_a_process_with_no_vocabulary_implies_nothing():
    from scone_memory.entities.read import load_projection

    engine = await memory()
    try:
        said = await engine.remember("default", "Ana Alves works at Meridian Health.")
        await engine.assert_fact("default", "ana alves", "works_at", "Meridian Health",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=said.episode_id,
                                 quote="Ana Alves works at Meridian Health.")
        found, _ = await load_projection(engine, "default", mode="current")
        assert not found.implied, [one.predicate for one in found.implied]
    finally:
        await engine.close()
