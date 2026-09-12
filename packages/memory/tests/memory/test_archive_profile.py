"""What an archive says it is, and what it refuses to carry.

An archive that does not name its own shape can only be read by guessing,
and a reader that guesses will one day drop something and call it a
success. This is the half of portability that is not about moving data:
saying what the format is, refusing a format from the future rather than
reading it hopefully, and refusing a record carrying something this
version does not understand rather than importing the part it recognises.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.memory.archive import ARCHIVE_PROFILE
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"
DAY = "2024-01-01T00:00:00Z"


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z")).open()


async def filled() -> MemoryEngine:
    engine = await memory()
    note = await engine.remember(SPACE, "Alice Chen joined Acme Robotics.")
    await engine.assert_fact(SPACE, "alice chen", "works_at", "Acme Robotics", valid_from=DAY,
                             source_episode_id=note.episode_id, quote="Alice Chen joined Acme Robotics")
    return engine


async def records(engine) -> list[dict]:
    return [record async for record in engine.export(SPACE)]


async def test_an_archive_says_what_it_is_before_it_says_anything_else():
    [header, *rest] = await records(await filled())
    assert header["type"] == "archive" and header["profile"] == ARCHIVE_PROFILE
    assert header["space"] == SPACE and header["wrote_at"]
    assert {record["type"] for record in rest} == {"episode", "fact"}


async def test_an_archive_from_a_later_version_is_refused_rather_than_read_hopefully():
    written = await records(await filled())
    written[0] = {**written[0], "profile": "scone.archive/99"}
    with pytest.raises(InvalidInput, match="scone.archive/99"):
        await (await memory()).import_records(SPACE, written)


async def test_an_archive_with_no_header_is_read_as_the_first_profile():
    """Archives written before the header existed are still archives, and
    the summary says which profile it read them as."""
    written = [record for record in await records(await filled()) if record["type"] != "archive"]
    summary = await (await memory()).import_records(SPACE, written)
    assert summary.profile == ARCHIVE_PROFILE and summary.episodes == 1 and summary.facts == 1


async def test_a_record_carrying_something_this_version_cannot_keep_is_refused():
    """The lossy case that matters: a field a later version added. Importing
    the part we recognise would look like a success and quietly drop it."""
    written = await records(await filled())
    written = [{**record, "attachments": [{"blob_id": "x"}]} if record["type"] == "episode" else record
               for record in written]
    with pytest.raises(InvalidInput, match="attachments"):
        await (await memory()).import_records(SPACE, written)


async def test_the_refusal_names_every_field_it_did_not_know():
    written = await records(await filled())
    written = [{**record, "weight": 1, "colour": "red"} if record["type"] == "fact" else record
               for record in written]
    with pytest.raises(InvalidInput, match="colour, weight"):
        await (await memory()).import_records(SPACE, written)


async def test_what_an_archive_carries_still_round_trips_whole():
    source = await filled()
    target = await memory()
    summary = await target.import_records(SPACE, await records(source))
    assert summary.episodes == 1 and summary.facts == 1 and summary.profile == ARCHIVE_PROFILE
    assert [fact.object for fact in await target.documents.list_facts(SPACE, include_closed=True)] == [
        "Acme Robotics"]


async def test_an_archive_says_what_this_profile_does_not_carry():
    """A space with attachments exports without them. That is a real loss,
    so the archive says it rather than letting a reader believe a dump of
    an illustrated space is the whole of it."""
    engine = await memory()
    kept = await engine.blobs.put(SPACE, b"\\x89PNG fake", "image/png", "shot.png")
    added = await engine.remember(SPACE, "The office, photographed.")
    await engine.blobs.link(SPACE, kept.attachment_id, added.episode_id)
    [header, *_] = await records(engine)
    assert header["carries"] == ["episodes", "facts", "fact_links", "affirmations"]
    assert header["not_carried"] == {"attachments": 1}


async def test_an_archive_of_a_space_with_nothing_left_behind_says_nothing_was():
    [header, *_] = await records(await filled())
    assert header["not_carried"] == {}
