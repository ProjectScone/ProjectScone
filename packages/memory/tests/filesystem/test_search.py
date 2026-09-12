"""Searching the tree: the same recall, answered as paths.

An agent that has been given a filesystem will try to grep it. This
answers that in the tree's own terms — a path, and the words that matched
— while the searching itself is the engine's ordinary recall, so there is
no second index and no second idea of what a good answer is.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.filesystem import FilesystemPolicy, MemoryFilesystem, PathRefused
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"


async def tree() -> MemoryFilesystem:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"),
                                events=InMemoryEventLog()).open()
    fs = MemoryFilesystem(engine, SPACE, FilesystemPolicy(writable=True))
    note = await engine.remember(SPACE, "The Lisbon office opened in March and holds forty desks.")
    await engine.assert_fact(SPACE, "lisbon office", "opened_on", "March 2024",
                             valid_from="2024-01-01T00:00:00Z", source_episode_id=note.episode_id,
                             quote="The Lisbon office opened in March")
    await fs.write("/notes/plan.md", "Ship the graph work on Friday, then the desks.")
    return fs


async def test_searching_answers_in_paths():
    fs = await tree()
    found = await fs.search("desks")
    assert found.hits, found
    assert {hit.path for hit in found.hits} <= {"/notes/plan.md", "/episodes/1.md", "/episodes/2.md",
                                                "/facts/lisbon%20office.md"}
    assert all(hit.excerpt for hit in found.hits)
    assert found.query == "desks"


async def test_a_note_is_found_at_its_note_path_not_at_its_episode():
    fs = await tree()
    found = await fs.search("ship the graph work")
    assert found.hits[0].path == "/notes/plan.md" and found.hits[0].of == "note"


async def test_a_search_can_be_held_to_one_directory():
    fs = await tree()
    found = await fs.search("desks", under="/notes")
    assert found.under == "/notes"
    assert found.hits and all(hit.path.startswith("/notes/") for hit in found.hits)


async def test_a_claim_is_found_at_the_page_of_its_subject():
    fs = await tree()
    found = await fs.search("when did the lisbon office open")
    assert any(hit.path == "/facts/lisbon%20office.md" and hit.of == "fact" for hit in found.hits), found.hits


async def test_a_search_that_names_a_directory_that_is_not_there_is_refused():
    fs = await tree()
    with pytest.raises(PathRefused, match="no such directory"):
        await fs.search("desks", under="/nowhere")


async def test_a_search_is_recorded_like_every_other_action():
    fs = await tree()
    await fs.search("desks")
    [event] = await fs.engine.events.query(SPACE, kind="filesystem.search", limit=10)
    # Every filesystem event carries the path it was about under one key,
    # so an audit can be read without knowing which action it was.
    assert event.payload["path"] == "/" and event.payload["hits"] >= 1
    assert "desks" not in str(event.payload), "what was searched for is not recorded by default"


async def test_a_search_is_bounded():
    fs = await tree()
    with pytest.raises(PathRefused, match="limit"):
        await fs.search("desks", limit=0)
