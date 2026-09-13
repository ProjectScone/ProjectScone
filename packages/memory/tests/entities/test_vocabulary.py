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


async def test_a_space_with_a_saved_vocabulary_beats_the_process(tmp_path):
    """The capability, end to end: what the space holds is what every
    reader applies, whatever each process was configured with."""
    from scone_memory.entities.vocabulary_store import VocabularyStore

    kept = VocabularyStore(tmp_path / "v.db", key=b"k" * 32)
    kept.save("default", RelationMeanings(symmetric=["married_to"]), expected_revision=0)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=EMPLOYS, vocabulary=kept).open()
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "space", held.why
        assert held.meanings is not None and held.meanings.reads_both_ways("married_to")
        assert held.meanings.opposite("works_at") is None, \
            "the process's configuration must not leak past the space's"
        assert "saved at" in held.why, held.why
    finally:
        await engine.close()
        kept.close()


async def test_a_cleared_space_falls_back_and_says_it_was_cleared(tmp_path):
    from scone_memory.entities.vocabulary_store import VocabularyStore

    kept = VocabularyStore(tmp_path / "v.db", key=b"k" * 32)
    kept.save("default", RelationMeanings(symmetric=["married_to"]), expected_revision=0)
    kept.clear("default", expected_revision=1)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=EMPLOYS, vocabulary=kept).open()
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "process" and "cleared" in held.why, held.why
        assert held.meanings is not None and held.meanings.opposite("works_at") == "employs"
    finally:
        await engine.close()
        kept.close()


async def test_a_saved_vocabulary_reaches_the_projection(tmp_path):
    """A store nothing reads would be a receipt, not a capability."""
    from scone_memory.entities.read import load_projection
    from scone_memory.entities.vocabulary_store import VocabularyStore

    kept = VocabularyStore(tmp_path / "v.db", key=b"k" * 32)
    kept.save("default", EMPLOYS, expected_revision=0)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                vocabulary=kept).open()
    try:
        said = await engine.remember("default", "Ana Alves works at Meridian Health.")
        await engine.assert_fact("default", "ana alves", "works_at", "Meridian Health",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=said.episode_id,
                                 quote="Ana Alves works at Meridian Health.")
        found, _ = await load_projection(engine, "default", mode="current")
        assert any(one.predicate == "employs" for one in found.implied), \
            "a process configured with nothing still reads the space's vocabulary"
    finally:
        await engine.close()
        kept.close()


def test_the_wire_carries_the_vocabulary_provenance(tmp_path):
    """In process is not on the wire. A caller who cannot see where the
    vocabulary came from cannot tell why two answers about one space
    differ, and a claim that every answer says so is untrue of an answer
    that does not carry it."""
    import asyncio

    from fastapi.testclient import TestClient

    from scone_memory.api import create_app
    from scone_memory.entities.vocabulary_store import VocabularyStore

    kept = VocabularyStore(tmp_path / "v.db", key=b"k" * 32)
    kept.save("default", EMPLOYS, expected_revision=0)

    async def built():
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                    HashEmbedder(), vocabulary=kept).open()
        said = await engine.remember("default", "Ana Alves works at Meridian Health.")
        await engine.assert_fact("default", "ana alves", "works_at", "Meridian Health",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=said.episode_id,
                                 quote="Ana Alves works at Meridian Health.")
        return engine

    engine = asyncio.run(built())
    try:
        with TestClient(create_app(engine, {"k": "default"})) as client:
            listed = client.get("/v1/entities", headers={"authorization": "Bearer k"})
            assert listed.status_code == 200, listed.text
            entities = listed.json()["entities"]
            assert entities, listed.text
            found = client.get(f"/v1/entities/{entities[0]['id']}",
                               headers={"authorization": "Bearer k"})
            assert found.status_code == 200, found.text
            # Both the listing and the detail view carry the provenance,
            # because the claim is about every answer built from a
            # projection rather than one route.
            for block in (listed.json()["coverage"], found.json()["coverage"]):
                assert block["vocabulary_source"] == "space", block
                assert "saved at" in block["vocabulary_why"], block["vocabulary_why"]
                assert "revision 1" in block["vocabulary_why"], block["vocabulary_why"]
    finally:
        asyncio.run(engine.close())
        kept.close()


def test_the_engine_imports_without_the_store_s_dependencies():
    """A base install promises pydantic alone. Importing the vocabulary
    store at module scope made `import scone_memory` require cryptography,
    which is a dependency regression rather than a feature."""
    import builtins
    import subprocess
    import sys

    script = (
        "import builtins\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name.startswith('cryptography'):\n"
        "        raise ImportError('absent in a base install')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "from scone_memory import MemoryEngine\n"
        "print('ok')\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr[-400:]


async def test_a_space_not_resolved_at_open_says_so_and_touches_no_store(tmp_path):
    """These stores are thread-bound, so a request must not reach one. A
    space nobody named at open therefore falls back -- and says it fell
    back, rather than quietly serving process configuration as though the
    space had nothing saved."""
    from scone_memory.entities.vocabulary_store import VocabularyStore

    kept = VocabularyStore(tmp_path / "v.db", key=b"k" * 32)
    kept.save("alpha", RelationMeanings(symmetric=["married_to"]), expected_revision=0)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                               relation_meanings=EMPLOYS, vocabulary=kept,
                               vocabulary_spaces=["default"]).open()
    reads = []
    original = kept.get
    kept.get = lambda space: (reads.append(space), original(space))[1]
    try:
        held = await read_vocabulary(engine, "alpha")
        assert reads == [], "an unresolved space must not reach the thread-bound store"
        assert held.source == "process", held.why
        assert "was not read when this engine opened" in held.why, held.why
    finally:
        kept.get = original
        await engine.close()
        kept.close()


async def test_the_vocabulary_handed_out_cannot_be_mutated_into_the_cache(tmp_path):
    """It was handed out by reference, so a caller could empty it and
    leave the cached projection implying edges the reported vocabulary no
    longer mentioned -- under the same identity."""
    from scone_memory.entities.vocabulary_store import VocabularyStore

    kept = VocabularyStore(tmp_path / "v.db", key=b"k" * 32)
    kept.save("default", EMPLOYS, expected_revision=0)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                vocabulary=kept).open()
    try:
        held = await read_vocabulary(engine, "default")
        assert held.meanings is not None
        with pytest.raises(Exception):
            held.meanings.inverse.clear()
        again = await read_vocabulary(engine, "default")
        assert again.meanings is not None
        assert again.meanings.opposite("works_at") == "employs", "the held vocabulary stands"
    finally:
        await engine.close()
        kept.close()
