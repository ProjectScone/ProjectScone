"""Memory as a tree of paths: what an agent may list, read, and never do.

The tree is a view of the ledger, not a second store. Every path resolves
to something the engine already holds, reading one changes nothing, and a
path that tries to leave the space it was opened for is refused rather
than resolved.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.filesystem import (
    MAX_FILE_BYTES,
    MAX_PATH,
    PathRefused,
    list_path,
    read_file,
)
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"


async def memory(**options) -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z"), **options).open()


async def filled() -> MemoryEngine:
    engine = await memory()
    note = await engine.remember(SPACE, "Alice Chen joined Acme Robotics in May 2021.")
    await engine.assert_fact(SPACE, "alice chen", "works_at", "Acme Robotics",
                             valid_from="2024-01-01T00:00:00Z", source_episode_id=note.episode_id,
                             quote="Alice Chen joined Acme Robotics")
    await engine.remember(SPACE, "The Lisbon office opened on a Tuesday.")
    return engine


async def test_the_root_says_what_a_space_holds():
    listing = await list_path(await filled(), SPACE, "/")
    assert [entry.path for entry in listing.entries] == ["/entities", "/episodes", "/facts", "/notes"]
    assert all(entry.kind == "directory" for entry in listing.entries)
    assert listing.total == 4 and not listing.truncated


async def test_an_episode_is_a_file_holding_exactly_what_was_stored():
    engine = await filled()
    listing = await list_path(engine, SPACE, "/episodes")
    assert [entry.path for entry in listing.entries] == ["/episodes/1.md", "/episodes/2.md"]
    page = await read_file(engine, SPACE, "/episodes/1.md")
    assert page.text == "Alice Chen joined Acme Robotics in May 2021."
    assert page.of == "episode" and page.bytes == len(page.text.encode())
    assert page.revision == await engine.documents.revision(SPACE)


async def test_the_claims_about_a_subject_read_as_a_page_that_cites_them():
    page = await read_file(await filled(), SPACE, "/facts/alice%20chen.md")
    assert "works_at" in page.text and "Acme Robotics" in page.text
    assert "fact 1" in page.text and page.of == "fact"


async def test_a_name_that_is_not_a_filename_survives_the_round_trip():
    engine = await memory()
    await engine.assert_fact(SPACE, "a/b c", "knows", "Bob", valid_from="2024-01-01T00:00:00Z")
    listing = await list_path(engine, SPACE, "/facts")
    [entry] = listing.entries
    assert entry.path == "/facts/a%2Fb%20c.md", entry.path
    page = await read_file(engine, SPACE, entry.path)
    assert "knows" in page.text


@pytest.mark.parametrize("path, says", [
    ("/episodes/../../etc/passwd", "leave"),
    ("//episodes", "one slash"),
    ("episodes/1.md", "start"),
    ("/episodes/\x01.md", "control"),
    ("/" + "a" * (MAX_PATH + 1), "long"),
    ("/nowhere", "no such"),
    ("/episodes/nine.md", "episode"),
])
async def test_a_path_that_cannot_mean_anything_here_is_refused(path, says):
    with pytest.raises(PathRefused, match=says):
        await read_file(await filled(), SPACE, path)


async def test_a_listing_is_bounded_and_says_what_it_left_out():
    engine = await memory()
    for n in range(40):
        await engine.remember(SPACE, f"note number {n}")
    listing = await list_path(engine, SPACE, "/episodes", limit=10)
    assert len(listing.entries) == 10 and listing.total == 40 and listing.truncated
    assert listing.next_offset == 10
    more = await list_path(engine, SPACE, "/episodes", limit=10, offset=10)
    assert more.entries[0].path == "/episodes/11.md"


async def test_a_file_longer_than_the_budget_is_cut_and_says_so():
    engine = await memory()
    await engine.remember(SPACE, "x" * (MAX_FILE_BYTES + 100))
    page = await read_file(engine, SPACE, "/episodes/1.md")
    assert page.truncated and page.bytes == MAX_FILE_BYTES
    whole = await read_file(engine, SPACE, "/episodes/1.md", max_bytes=MAX_FILE_BYTES + 1000)
    assert not whole.truncated


async def test_reading_changes_nothing():
    engine = await filled()
    before = await engine.documents.revision(SPACE)
    await list_path(engine, SPACE, "/episodes")
    await read_file(engine, SPACE, "/episodes/1.md")
    assert await engine.documents.revision(SPACE) == before


async def test_a_space_with_more_episodes_than_the_tree_reads_says_so(monkeypatch):
    """The listing reads a bounded number of episodes. It must not then
    report that bound as though it were how many there are: a reader who
    pages to the end would think they had seen everything."""
    from scone_memory import filesystem as tree

    monkeypatch.setattr(tree, "MAX_LISTED", 3)
    engine = await memory()
    for n in range(7):
        await engine.remember(SPACE, f"note number {n}")
    listing = await list_path(engine, SPACE, "/episodes", limit=10)
    assert listing.total == 7, "how many there are, not how many were read"
    assert len(listing.entries) == 3 and listing.truncated
    assert listing.capped == 3, "and how many the tree would read"
