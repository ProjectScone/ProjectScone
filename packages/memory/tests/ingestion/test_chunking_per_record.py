"""A caller can choose how one record is cut, and the receipt says how it was.

Three ways to cut prose exist -- by length, at the structure it carries,
where its subject changes -- and code has a fourth. Until now the choice
was an engine-construction switch: every record in a process cut the
same way, nothing on the command line or the HTTP route could ask for
another, and the structure chunker's own receipt (what landed on a
boundary, what was split by size, whether the unit bound bit) was thrown
away at the batch. Cut positions decide what chunks exist and stored
offsets are part of the shared specification, so the engine default
stays; this is the per-record choice, named, with its receipt, kept
where a recovery can read it back.
"""
from __future__ import annotations

import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.chunker import chunk_spans
from scone_memory.ingestion.structure_chunks import structured_spans

PARA = ("The yard kept the crane running through the winter, and the log says so in the plain "
        "way logs do: dates, names, the odd complaint about rust. ") * 6
DOC = "# Handover\n\n" + PARA + "\n\n## The complaint\n\n" + PARA + "\n\n## The settlement\n\n" + PARA
CODE = "import os\n\n\ndef alpha():\n    return os.getcwd()\n\n\nclass Beta:\n    pass\n"


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield memory
    await memory.close()


async def texts(engine, episode_id):
    return [chunk.text for chunk in await engine.documents.chunks_of("default", episode_id)]


def by_length(content, target):
    return [content[s.start:s.end] for s in chunk_spans(content, target)]


def by_structure(content, target):
    return [content[s.start:s.end] for s in structured_spans(content, target).spans]


async def test_structure_can_be_chosen_for_one_record_and_the_receipt_says_so(engine):
    added = await engine.remember("default", DOC, source="handover.txt", chunking="structure")
    assert await texts(engine, added.episode_id) == by_structure(DOC, engine.chunk_target)
    assert added.chunking == "structure"
    assert added.structure is not None and added.structure["at_boundary"] >= 2 and added.structure["capped"] is False
    assert set(added.structure) == {"units", "at_boundary", "by_size", "tables", "over_target", "capped", "why"}
    assert (await engine.episode("default", added.episode_id)).metadata["chunking"] == "structure"


async def test_the_default_cut_is_unchanged_and_named(engine):
    added = await engine.remember("default", DOC, source="handover.txt")
    assert await texts(engine, added.episode_id) == by_length(DOC, engine.chunk_target)
    assert added.chunking == "length" and added.structure is None
    assert "chunking" not in (await engine.episode("default", added.episode_id)).metadata


async def test_code_is_cut_as_code_by_default_and_says_so(engine):
    added = await engine.remember("default", CODE, kind="file", source="pkg/a.py")
    assert added.chunking == "code" and added.structure is None


async def test_length_can_be_insisted_on_for_a_code_source(engine):
    added = await engine.remember("default", CODE, kind="file", source="pkg/a.py", chunking="length")
    assert added.chunking == "length"
    assert await texts(engine, added.episode_id) == by_length(CODE, engine.chunk_target)


async def test_code_cutting_needs_a_source_that_names_its_language(engine):
    with pytest.raises(InvalidInput):
        await engine.remember("default", DOC, source="handover.txt", chunking="code")
    assert (await engine.status("default")).episodes == 0


async def test_an_unknown_mode_is_refused_before_anything_is_stored(engine):
    with pytest.raises(InvalidInput):
        await engine.remember("default", DOC, chunking="clever")
    assert (await engine.status("default")).episodes == 0


async def test_a_conflicting_chunking_key_in_metadata_is_refused(engine):
    with pytest.raises(InvalidInput):
        await engine.remember("default", DOC, chunking="structure", metadata={"chunking": "length"})


async def test_semantic_can_be_chosen_per_record(engine):
    added = await engine.remember("default", DOC, source="handover.txt", chunking="semantic")
    assert added.chunking == "semantic" and added.structure is None
    assert (await engine.episode("default", added.episode_id)).metadata["chunking"] == "semantic"


async def test_recovery_after_a_crash_cuts_the_way_the_record_asked(engine):
    """A mode chosen for one record must outlive an interruption between
    storing the episode and cutting it. An ordinary exception is rolled
    back by the batch itself; a crash is not, and leaves exactly this: the
    episode row, its inflight mark, and no chunks. Recovery reads the mode
    back from the episode rather than falling to the engine's default."""
    from scone_memory.ingestion.batch import validated_record
    from scone_memory.ingestion.records import Record

    new = validated_record("default", Record(DOC, source="handover.txt", chunking="structure"), engine.clock())
    assert new.metadata["chunking"] == "structure"
    await engine.documents.mark_inflight("default", new.content_hash)
    episode = await engine.documents.insert_episode(new)
    report = await engine.recover()
    assert report.completed == 1, report
    assert await texts(engine, episode.episode_id) == by_structure(DOC, engine.chunk_target)


async def test_over_http_the_mode_is_a_field_and_the_receipt_carries_it(engine):
    from httpx import ASGITransport, AsyncClient

    from scone_memory.api import create_app

    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {"k": "default"})), base_url="http://fixture") as client:
        auth = {"authorization": "Bearer k"}
        one = await client.post("/v1/episodes", json={"content": DOC, "source": "handover.txt", "chunking": "structure"}, headers=auth)
        assert one.status_code == 200, one.text
        assert one.json()["chunking"] == "structure" and one.json()["structure"]["at_boundary"] >= 2
        bad = await client.post("/v1/episodes", json={"content": DOC, "chunking": "clever"}, headers=auth)
        assert bad.status_code == 422
        batch = await client.post("/v1/episodes/batch", json={"records": [
            {"content": DOC + "\n\nMore.", "source": "b.txt", "chunking": "structure"},
            {"content": "plain " + PARA}]}, headers=auth)
        assert batch.status_code == 200, batch.text
        # The batch answers with ids and outcomes; the mode each record asked
        # for is on its episode, where a recovery reads it.
        structured = await engine.episodes("default", {"chunking": "structure"})
        assert sorted(episode.source or "" for episode in structured) == ["b.txt", "handover.txt"]
        assert await texts(engine, next(e.episode_id for e in structured if e.source == "b.txt")) == by_structure(DOC + "\n\nMore.", engine.chunk_target)


async def test_from_the_command_line(engine, tmp_path):
    import io

    from scone_memory.runtime.cli import build_parser, run

    path = tmp_path / "handover.txt"
    path.write_text(DOC)
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "remember", str(path), "--source", "handover.txt", "--chunking", "structure"]),
                     engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    first = json.loads(out.getvalue().splitlines()[0])
    receipt = first[0] if isinstance(first, list) else first
    assert receipt["chunking"] == "structure" and receipt["structure"]["at_boundary"] >= 2
