"""A record chooses a chunking profile the way it chooses chunking.

The profile is a parameter of structure chunking, so it is its own field
rather than a fifth chunking mode: ``chunking`` stays the closed set of
four ways to cut, and ``chunking_profile`` names the boundaries one of
them cuts at. It is kept on the episode's metadata beside ``chunking`` so
a recovery re-cuts with it, it is refused before anything is stored when
it names nothing, and the receipt says which profile cut and what its
rules matched.
"""
from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.chunking_profiles import profiled_spans
from scone_memory.ingestion.records import Record

STATUTE = ("Article 1 Scope\n" + "This Act applies to every record kept by a public body. " * 8 + "\n\n"
           "Article 2\nDefinitions\n\n"
           + "".join(f"({letter}) '{letter}-term' means a thing the Act names. " + "It is defined here. " * 12 + "\n"
                     for letter in "abc"))
QA = "Q: How do I reset my password?\nA: Open Settings. " + "Then follow the link. " * 25 + "\n\nQ: Again?\nA: Yes.\n"


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield memory
    await memory.close()


async def texts(engine, episode_id):
    return [chunk.text for chunk in await engine.documents.chunks_of("default", episode_id)]


def by_profile(content, profile, target):
    return [content[s.start:s.end] for s in profiled_spans(content, target, profile=profile).spans]


async def test_a_profile_cuts_the_record_and_the_receipt_says_which_and_what_it_matched(engine):
    added = await engine.remember("default", STATUTE, source="act.txt", chunking_profile="statute")
    assert await texts(engine, added.episode_id) == by_profile(STATUTE, "statute", engine.chunk_target)
    assert added.chunking == "structure"
    assert added.structure is not None and added.structure["profile"] == "statute"
    assert added.structure["matched"] == {"article": 2, "letter": 3}
    assert sum(added.structure["began"].values()) == added.structure["at_boundary"]
    metadata = (await engine.episode("default", added.episode_id)).metadata
    assert metadata["chunking"] == "structure" and metadata["chunking_profile"] == "statute"


async def test_a_profile_wins_over_code_cutting_for_a_code_source(engine):
    added = await engine.remember("default", QA, source="faq.py", chunking_profile="qa")
    assert added.chunking == "structure" and added.structure["profile"] == "qa"


async def test_structure_may_be_named_alongside_the_profile(engine):
    added = await engine.remember("default", QA, chunking="structure", chunking_profile="qa")
    assert added.structure["profile"] == "qa"


async def test_an_unknown_profile_is_refused_before_anything_is_stored(engine):
    with pytest.raises(InvalidInput, match="statute"):
        await engine.remember("default", STATUTE, chunking_profile="sonnet")
    assert (await engine.status("default")).episodes == 0


async def test_an_unknown_profile_is_refused_even_for_a_duplicate(engine):
    await engine.remember("default", STATUTE)
    with pytest.raises(InvalidInput):
        await engine.remember("default", STATUTE, chunking_profile="sonnet")


async def test_a_profile_that_is_not_text_is_refused_not_crashed_on(engine):
    record = Record.from_dict({"content": STATUTE, "chunking_profile": ["statute"]})
    with pytest.raises(InvalidInput):
        await engine.remember_many("default", [record])


async def test_a_partial_batch_answers_an_unknown_profile_for_that_record_alone(engine):
    added = await engine.remember_many("default", [Record(STATUTE, chunking_profile="sonnet"), Record(QA, chunking_profile="qa")],
                                       partial=True)
    assert added[0].outcome == "failed" and "chunking_profile" in (added[0].reason or "")
    assert added[1].outcome == "accepted"
    assert (await engine.status("default")).episodes == 1


async def test_a_profile_with_another_chunking_is_refused(engine):
    with pytest.raises(InvalidInput, match="structure"):
        await engine.remember("default", STATUTE, chunking="length", chunking_profile="statute")
    assert (await engine.status("default")).episodes == 0


async def test_a_conflicting_profile_in_metadata_is_refused(engine):
    with pytest.raises(InvalidInput):
        await engine.remember("default", STATUTE, chunking_profile="statute", metadata={"chunking_profile": "qa"})


async def test_a_profile_held_in_metadata_alone_is_honoured(engine):
    """An archive import carries metadata, not the field."""
    added = await engine.remember("default", QA, metadata={"chunking_profile": "qa"})
    assert added.structure is not None and added.structure["profile"] == "qa"
    assert (await engine.episode("default", added.episode_id)).metadata["chunking"] == "structure"


async def test_recovery_after_a_crash_cuts_with_the_same_profile(engine):
    from scone_memory.ingestion.batch import validated_record

    new = validated_record("default", Record(QA, chunking_profile="qa"), engine.clock())
    await engine.documents.mark_inflight("default", new.content_hash)
    episode = await engine.documents.insert_episode(new)
    report = await engine.recover()
    assert report.completed == 1 and report.rechunked == 1, report
    assert await texts(engine, episode.episode_id) == by_profile(QA, "qa", engine.chunk_target)


async def test_over_http_the_profile_is_a_field_on_one_record_and_on_each_in_a_batch(engine):
    from httpx import ASGITransport, AsyncClient

    from scone_memory.api import create_app

    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {"k": "default"})), base_url="http://fixture") as client:
        auth = {"authorization": "Bearer k"}
        one = await client.post("/v1/episodes", json={"content": STATUTE, "chunking_profile": "statute"}, headers=auth)
        assert one.status_code == 200, one.text
        assert one.json()["structure"]["profile"] == "statute"
        bad = await client.post("/v1/episodes", json={"content": QA, "chunking_profile": "sonnet"}, headers=auth)
        assert bad.status_code == 422 and "chunking_profile" in bad.text
        batch = await client.post("/v1/episodes/batch", json={"records": [
            {"content": QA, "chunking_profile": "qa"}, {"content": "plain " + QA}]}, headers=auth)
        assert batch.status_code == 200, batch.text
        [held] = await engine.episodes("default", {"chunking_profile": "qa"})
        assert await texts(engine, held.episode_id) == by_profile(QA, "qa", engine.chunk_target)
    assert (await engine.status("default")).episodes == 3


async def test_from_the_command_line(engine, tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    path = tmp_path / "act.txt"
    path.write_text(STATUTE)
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "remember", str(path), "--chunking-profile", "statute"]),
                     engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    first = json.loads(out.getvalue().splitlines()[0])
    receipt = first[0] if isinstance(first, list) else first
    assert receipt["structure"]["profile"] == "statute"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["remember", str(path), "--chunking-profile", "sonnet"])


async def test_a_jsonl_record_names_its_profile(engine, tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"content": QA, "chunking_profile": "qa"}) + "\n")
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "remember", str(path), "--jsonl"]), engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    [held] = await engine.episodes("default", {"chunking_profile": "qa"})
    assert await texts(engine, held.episode_id) == by_profile(QA, "qa", engine.chunk_target)
