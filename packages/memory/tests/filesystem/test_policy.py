"""What an agent may do with the tree, and what is written down when it does.

A filesystem an agent can write to is a filesystem that can quietly
become a second place where things are true. This one cannot: a note is
an ordinary episode with the path as its source, so it is recalled,
distilled, forgotten and exported like anything else, and the tree is
still only a view.

Nothing is writable unless the owner says so, writes land in one place
and nowhere else, a write that would land on top of somebody else's is
refused rather than resolved, and every action — including every refusal
— is recorded where the space's other events are.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.filesystem import (
    FilesystemPolicy,
    MAX_NOTE_BYTES,
    MemoryFilesystem,
    PathConflict,
    PathRefused,
)
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"


async def tree(writable: bool = True, **options) -> MemoryFilesystem:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"),
                                events=InMemoryEventLog(), **options).open()
    return MemoryFilesystem(engine, SPACE, FilesystemPolicy(writable=writable))


async def kinds(fs: MemoryFilesystem) -> list[str]:
    return [event.kind for event in await fs.engine.events.query(SPACE, limit=50)]


async def test_a_note_written_through_the_tree_is_an_ordinary_episode():
    """It is recalled, cited and forgotten like anything else, because it
    is not a second kind of thing."""
    fs = await tree()
    written = await fs.write("/notes/plan.md", "Ship the graph work on Friday.")
    assert written.path == "/notes/plan.md" and written.episode_id == 1
    page = await fs.read("/notes/plan.md")
    assert page.text == "Ship the graph work on Friday." and page.of == "note"
    same = await fs.read("/episodes/1.md")
    assert same.text == page.text, "one thing, two ways of looking at it"
    found = await fs.engine.recall(SPACE, "ship the graph work")
    assert found.items and found.items[0].source == "fs:/notes/plan.md"


async def test_a_tree_that_was_not_made_writable_writes_nothing():
    fs = await tree(writable=False)
    with pytest.raises(PathRefused, match="read only"):
        await fs.write("/notes/plan.md", "anything")
    assert await fs.engine.documents.counts(SPACE) == await fs.engine.documents.counts(SPACE)
    assert "filesystem.refused" in await kinds(fs)


async def test_a_view_of_the_ledger_is_not_a_place_to_write():
    fs = await tree()
    for path in ("/episodes/1.md", "/facts/alice.md", "/entities/alice.md", "/notes"):
        with pytest.raises(PathRefused, match="notes"):
            await fs.write(path, "no")


async def test_writing_a_note_again_supersedes_it_and_keeps_what_was_there():
    fs = await tree()
    first = await fs.write("/notes/plan.md", "Ship on Friday.")
    second = await fs.write("/notes/plan.md", "Ship on Monday.", if_version=(await fs.read("/notes/plan.md")).version)
    assert second.episode_id != first.episode_id
    assert (await fs.read("/notes/plan.md")).text == "Ship on Monday."
    old = await fs.read(f"/episodes/{first.episode_id}.md")
    assert old.text == "Ship on Friday.", "nothing stored is rewritten"


async def test_a_write_onto_a_note_that_moved_under_it_is_refused():
    """Two writers, one note: the second read an older version, so its
    write is refused rather than quietly landing on top."""
    fs = await tree()
    await fs.write("/notes/plan.md", "Ship on Friday.")
    stale = (await fs.read("/notes/plan.md")).version
    await fs.write("/notes/plan.md", "Ship on Monday.", if_version=stale)
    with pytest.raises(PathConflict, match="moved"):
        await fs.write("/notes/plan.md", "Ship on Tuesday.", if_version=stale)


async def test_a_write_is_not_refused_because_somebody_else_wrote_something_else():
    """A note's version is the note's own. A space revision would move
    whenever anything at all was written, and refuse a writer who had
    conflicted with nobody."""
    fs = await tree()
    await fs.write("/notes/plan.md", "Ship on Friday.")
    mine = (await fs.read("/notes/plan.md")).version
    for n in range(3):
        await fs.engine.remember(SPACE, f"something else entirely {n}")
    await fs.write("/notes/plan.md", "Ship on Monday.", if_version=mine)
    assert (await fs.read("/notes/plan.md")).text == "Ship on Monday."


async def test_a_note_larger_than_the_policy_allows_is_refused():
    fs = await tree()
    with pytest.raises(PathRefused, match="at most"):
        await fs.write("/notes/plan.md", "x" * (MAX_NOTE_BYTES + 1))


async def test_every_action_is_written_where_the_space_keeps_its_events():
    fs = await tree()
    await fs.write("/notes/plan.md", "Ship on Friday.")
    await fs.list("/notes")
    await fs.read("/notes/plan.md")
    with pytest.raises(PathRefused):
        await fs.read("/nowhere/x.md")
    said = await kinds(fs)
    assert [kind for kind in said if kind.startswith("filesystem.")] == [
        "filesystem.refused", "filesystem.read", "filesystem.list", "filesystem.write"]
    assert "remember" in said, "a note is an episode, and says so in the space's events too"
    [refusal] = await fs.engine.events.query(SPACE, kind="filesystem.refused", limit=50)
    assert refusal.payload["path"] == "/nowhere/x.md" and refusal.payload["why"]


async def test_notes_are_listed_as_their_own_directory():
    fs = await tree()
    await fs.write("/notes/plan.md", "one")
    await fs.write("/notes/ideas/later.md", "two")
    listing = await fs.list("/notes")
    assert [entry.path for entry in listing.entries] == ["/notes/ideas/later.md", "/notes/plan.md"]
    assert all(entry.of == "note" for entry in listing.entries)
    root = await fs.list("/")
    assert "/notes" in [entry.path for entry in root.entries]
