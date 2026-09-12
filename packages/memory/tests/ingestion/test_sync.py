"""Keeping a space in step with a directory, including what left it.

`map` remembers files and notices when it has seen one before. What it
cannot do is notice that a file *changed* or that a file is *gone*, and
those are the two things that make the difference between an import and
a sync. The dangerous one is removal: forgetting memory because a file
is missing is destructive, so it is opt-in, previewed first, and refused
outright when the directory came back empty.
"""

from __future__ import annotations

import asyncio
import os
import pathlib

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog,
                           InMemoryVectorIndex, MemoryEngine)
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.sync import MAX_FILES, sync_directory

pytestmark = pytest.mark.asyncio


async def memory():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              events=InMemoryEventLog()).open()


def tree(root, **files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


async def count(engine, space="default"):
    return (await engine.status(space)).episodes


async def test_a_first_sync_adds_every_file(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass", "b/c.md": "# notes"})
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True)
        assert (done.added, done.updated, done.unchanged, done.removed) == (2, 0, 0, 0), done.record()
        assert await count(engine) == 2
    finally:
        await engine.close()


async def test_a_second_sync_of_the_same_files_writes_nothing(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        before = await engine.revision("default")
        again = await sync_directory(engine, "default", tmp_path, apply=True)
        assert (again.added, again.updated, again.unchanged) == (0, 0, 1), again.record()
        assert await engine.revision("default") == before, "an unchanged file is not a write"
    finally:
        await engine.close()


async def test_a_changed_file_is_an_update_not_a_second_memory(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        tree(tmp_path, **{"a.py": "def a(): return 1"})
        done = await sync_directory(engine, "default", tmp_path, apply=True)
        assert (done.added, done.updated, done.unchanged) == (0, 1, 0), done.record()
        held = await engine.episodes("default", {"sync": str(tmp_path.resolve())})
        assert len(held) == 1 and held[0].content == "def a(): return 1"
    finally:
        await engine.close()


async def test_a_missing_file_is_reported_but_not_forgotten_unless_asked(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass", "b.py": "def b(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        (tmp_path / "b.py").unlink()
        seen = await sync_directory(engine, "default", tmp_path, apply=True)
        assert seen.removed == 1 and seen.forgotten == 0, seen.record()
        assert await count(engine) == 2, "nothing is deleted without being asked"
        assert "not removed" in seen.text(), seen.text()
        gone = await sync_directory(engine, "default", tmp_path, apply=True, remove=True)
        assert gone.forgotten == 1 and await count(engine) == 1
    finally:
        await engine.close()


async def test_a_preview_changes_nothing_at_all(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass"})
    engine = await memory()
    try:
        planned = await sync_directory(engine, "default", tmp_path)
        assert planned.added == 1 and not planned.applied
        assert await count(engine) == 0, "a preview is a preview"
        assert "would" in planned.text(), planned.text()
    finally:
        await engine.close()


async def test_an_empty_directory_cannot_wipe_a_space(tmp_path):
    """The footgun this guards: pointing a sync at an unmounted or wrong
    directory and forgetting everything it had remembered."""
    tree(tmp_path, **{"a.py": "def a(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        (tmp_path / "a.py").unlink()
        with pytest.raises(InvalidInput) as refused:
            await sync_directory(engine, "default", tmp_path, apply=True, remove=True)
        assert "no files" in str(refused.value) and "1" in str(refused.value)
        assert await count(engine) == 1, "the refusal has to happen before anything is forgotten"
    finally:
        await engine.close()


async def test_what_is_there_is_counted_apart_from_what_was_read(tmp_path):
    tree(tmp_path, **{f"f{n}.py": f"def f{n}(): pass" for n in range(5)})
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True, limit=2)
        assert done.files_found == 5 and done.files_read == 2 and done.capped
        assert "5" in done.text() and "2" in done.text(), done.text()
        assert done.removed == 0, "the three files we did not read are not missing files"
    finally:
        await engine.close()


async def test_a_file_that_cannot_be_decoded_is_counted_apart_from_an_empty_one(tmp_path):
    (tmp_path / "good.py").write_text("def a(): pass", encoding="utf-8")
    (tmp_path / "empty.py").write_text("   \n", encoding="utf-8")
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00\x01 def a(): pass")
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True)
        assert done.empty == 1, done.record()
        assert done.added == 2, "replacement characters are still text worth keeping"
    finally:
        await engine.close()


async def test_the_sync_is_recorded_so_the_last_success_can_be_seen(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass"})
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True, marker="repo")
        [event] = await engine.events.query("default", kind="sync.completed")
        assert event.payload["marker"] == "repo" and event.payload["added"] == 1
        assert event.payload["applied"] is True and done.marker == "repo"
        planned = await sync_directory(engine, "default", tmp_path, marker="repo")
        assert len(await engine.events.query("default", kind="sync.completed")) == 1, \
            "a preview did nothing, so it did not succeed at anything"
        assert planned.unchanged == 1
    finally:
        await engine.close()


async def test_two_directories_in_one_space_do_not_delete_each_other(tmp_path):
    left = tree(tmp_path / "left", **{"a.py": "def a(): pass"})
    right = tree(tmp_path / "right", **{"b.py": "def b(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", left, apply=True, marker="left")
        await sync_directory(engine, "default", right, apply=True, marker="right")
        again = await sync_directory(engine, "default", left, apply=True, remove=True, marker="left")
        assert (again.unchanged, again.removed, again.forgotten) == (1, 0, 0), again.record()
        assert await count(engine) == 2
    finally:
        await engine.close()


async def test_a_root_that_is_not_a_directory_is_refused(tmp_path):
    engine = await memory()
    try:
        with pytest.raises(InvalidInput):
            await sync_directory(engine, "default", tmp_path / "nowhere", apply=True)
        with pytest.raises(InvalidInput):
            await sync_directory(engine, "default", tmp_path, apply=True, limit=MAX_FILES + 1)
    finally:
        await engine.close()


async def test_a_capped_walk_says_it_did_not_look_for_missing_files(tmp_path):
    """`removed: 0` from a walk that stopped early would read as "nothing is
    gone" when the truth is "we did not look". A file beyond the cap is
    unread, not missing, and forgetting it would be a real deletion caused
    by a limit."""
    tree(tmp_path, **{f"f{n}.py": f"def f{n}(): pass" for n in range(5)})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        capped = await sync_directory(engine, "default", tmp_path, apply=True, remove=True, limit=2)
        assert capped.removed == 0 and capped.forgotten == 0
        assert not capped.checked_for_missing
        assert "did not look" in capped.text(), capped.text()
        assert await count(engine) == 5, "a limit must never delete memory"
        whole = await sync_directory(engine, "default", tmp_path, apply=True, remove=True)
        assert whole.checked_for_missing and whole.removed == 0
    finally:
        await engine.close()


async def test_when_a_directory_was_last_synced_can_be_asked(tmp_path):
    from scone_memory.ingestion.sync import last_sync

    tree(tmp_path, **{"a.py": "def a(): pass"})
    engine = await memory()
    try:
        assert await last_sync(engine, "default", "repo") is None, "never synced is not an empty sync"
        await sync_directory(engine, "default", tmp_path, apply=True, marker="repo")
        await sync_directory(engine, "default", tmp_path, apply=True, marker="other")
        last = await last_sync(engine, "default", "repo")
        assert last is not None and last["marker"] == "repo" and last["added"] == 1
    finally:
        await engine.close()


async def test_a_store_with_no_event_log_says_it_cannot_tell(tmp_path):
    """Not every store keeps events, and "no log" must not look like
    "never synced"."""
    from scone_memory.ingestion.sync import NoEventLog, last_sync

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        tree(tmp_path, **{"a.py": "def a(): pass"})
        assert (await sync_directory(engine, "default", tmp_path, apply=True)).added == 1
        with pytest.raises(NoEventLog):
            await last_sync(engine, "default", "repo")
    finally:
        await engine.close()


async def test_a_root_inside_a_hidden_directory_is_read_not_skipped_whole(tmp_path):
    """The filter is meant to skip `.git` *under* the root. Applied to the
    whole path it skips the entire tree whenever the root itself is reached
    through a dot-segment -- `~/.config/notes`, `~/.claude/projects`, and
    this very worktree. `files_found` then reads 0 with the files on disk."""
    root = tmp_path / ".cache" / "repo"
    tree(root, **{"a.md": "# notes", "b/c.md": "# more"})
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", root, apply=True, marker="repo")
        assert done.files_found == 2 and done.added == 2, done.record()
    finally:
        await engine.close()


async def test_a_hidden_directory_under_the_root_is_still_skipped(tmp_path):
    """The rule the filter was for, which must survive the fix."""
    tree(tmp_path, **{"a.md": "# notes", ".git/config.md": "# not source",
                      "__pycache__/x.md": "# not source"})
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True)
        assert done.files_found == 1 and done.added == 1, done.record()
    finally:
        await engine.close()


async def test_the_same_filename_under_two_markers_is_two_memories(tmp_path):
    """Two directories synced into one space both holding `README.md` must
    not be one memory that each sync steals from the other. The identity a
    sync writes has to carry its marker, or the second sync forgets the
    first's file without being asked and without saying so."""
    left = tree(tmp_path / "left", **{"README.md": "the left project"})
    right = tree(tmp_path / "right", **{"README.md": "the right project"})
    engine = await memory()
    try:
        first = await sync_directory(engine, "default", left, apply=True, marker="left")
        second = await sync_directory(engine, "default", right, apply=True, marker="right")
        assert (first.added, second.added) == (1, 1), (first.record(), second.record())
        assert await count(engine) == 2, "neither sync may forget the other's file"
        held = {e.content for e in await engine.episodes("default", {"sync": "left"})}
        assert held == {"the left project"}, held
    finally:
        await engine.close()


async def test_narrowing_the_suffixes_does_not_report_the_rest_as_gone(tmp_path):
    """A file that dropped out of the walk is out of scope, not missing. The
    file cap was one way to stop seeing a file and it was handled; this is
    another, and forgetting on it would delete memory because a flag
    changed."""
    tree(tmp_path, **{"a.md": "# notes", "b.py": "def b(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        narrow = await sync_directory(engine, "default", tmp_path, apply=True,
                                      remove=True, suffixes=(".md",))
        assert narrow.removed == 0 and narrow.forgotten == 0, narrow.record()
        assert narrow.out_of_scope == 1, narrow.record()
        assert "out of scope" in narrow.text(), narrow.text()
        assert await count(engine) == 2, "a narrowed suffix list must never delete memory"
    finally:
        await engine.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a directory with no permissions")
async def test_a_directory_it_cannot_read_is_not_evidence_that_files_are_gone(tmp_path):
    """`rglob` swallows a PermissionError and returns what it could reach,
    which looks exactly like a smaller directory. Forgetting on that
    deletes memory for files that are still on disk -- the third way a file
    can leave the walk without leaving the disk, after the file cap and the
    suffix list."""
    tree(tmp_path, **{"open.py": "def a(): pass", "shut/present.py": "def b(): pass"})
    engine = await memory()
    try:
        await sync_directory(engine, "default", tmp_path, apply=True)
        assert await count(engine) == 2
        (tmp_path / "shut").chmod(0o000)
        try:
            blind = await sync_directory(engine, "default", tmp_path, apply=True, remove=True)
        finally:
            (tmp_path / "shut").chmod(0o755)
        assert blind.unreadable == 1, blind.record()
        assert not blind.checked_for_missing, blind.record()
        assert blind.forgotten == 0 and blind.removed == 0, blind.record()
        assert await count(engine) == 2, "a directory we could not read must never delete memory"
        assert "could not be read" in blind.text(), blind.text()
    finally:
        await engine.close()


async def test_a_readable_directory_reports_nothing_unreadable(tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass", "deep/b.py": "def b(): pass"})
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True, remove=True)
        assert done.unreadable == 0 and done.checked_for_missing, done.record()
        assert done.files_found == 2, done.record()
    finally:
        await engine.close()


async def test_a_source_this_sync_does_not_own_is_never_replaced(tmp_path):
    """Another writer's episode for the same path must survive. The space is
    shared -- a hand-written record, or one managed by the document
    DirectorySync -- and a sync that replaced whatever sat under a bare
    path would take those with it."""
    from scone_memory.ingestion.records import Record

    tree(tmp_path, **{"README.md": "the synced copy"})
    engine = await memory()
    try:
        await engine.replace("default", Record(content="somebody else's record",
                                               kind="file", source="README.md",
                                               dedup_key="README.md"))
        done = await sync_directory(engine, "default", tmp_path, apply=True, marker="mine")
        assert done.added == 1, done.record()
        assert await count(engine) == 2, "the other writer's record still stands"
        held = {e.content for e in await engine.episodes("default", {"sync": "mine"})}
        assert held == {"the synced copy"}, held
    finally:
        await engine.close()


async def test_the_key_a_sync_writes_is_namespaced_to_syncs(tmp_path):
    """Identity by construction rather than by convention: no hand-written
    key and no other service's key can collide with one of these unless it
    deliberately writes this namespace."""
    from scone_memory.ingestion.sync import _key

    assert _key("mine", "README.md").startswith("scone.sync/1:")
    assert _key("ab", "x") != _key("a", "b/x"), "a marker and a path must not be readable two ways"


async def test_a_symlink_does_not_pull_in_a_file_from_outside_the_root(tmp_path):
    """The root is the whole scope. A link is a file in it, but what it
    points at is not, and following one silently stores content the caller
    never pointed at -- and stores it under a path inside the root, so
    nothing in the space says where it really came from."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("not part of this project", encoding="utf-8")
    root = tmp_path / "root"
    tree(root, **{"real.md": "the real note"})
    (root / "link.md").symlink_to(outside / "secret.md")
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", root, apply=True)
        assert done.files_found == 1 and done.added == 1, done.record()
        assert done.links == 1 and "link" in done.text(), done.text()
        held = {e.content for e in await engine.episodes("default", {"sync": str(root.resolve())})}
        assert held == {"the real note"}, held
    finally:
        await engine.close()


async def test_a_linked_directory_is_not_walked(tmp_path):
    outside = tmp_path / "elsewhere"
    (outside / "deep").mkdir(parents=True)
    (outside / "deep" / "a.md").write_text("elsewhere entirely", encoding="utf-8")
    root = tmp_path / "root"
    tree(root, **{"real.md": "the real note"})
    (root / "linked").symlink_to(outside, target_is_directory=True)
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", root, apply=True)
        assert done.files_found == 1 and done.links == 1, done.record()
    finally:
        await engine.close()


async def test_a_long_file_is_not_read_whole_just_to_throw_most_of_it_away(tmp_path, monkeypatch):
    """A byte limit is a limit on reading, not only on storing. Reading a
    gigabyte to keep a kilobyte is how a sync of a large tree exhausts a
    machine, so this watches how much was actually asked for."""

    class Watched:
        """A handle that records the size of every read asked of it."""

        def __init__(self, handle, asked):
            self._handle, self._asked = handle, asked

        def read(self, size=-1):
            self._asked.append(size)
            return self._handle.read(size)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *gone):
            return self._handle.__exit__(*gone)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    (tmp_path / "big.md").write_bytes(b"x" * 262_144)
    asked: list[int] = []
    opening = pathlib.Path.open
    monkeypatch.setattr(pathlib.Path, "open",
                        lambda self, *a, **k: Watched(opening(self, *a, **k), asked))
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True, max_bytes=64)
    finally:
        await engine.close()
    assert done.cut == 1 and done.added == 1, done.record()
    assert asked == [65], f"one read of the limit plus a byte, not {asked}"


async def test_a_file_exactly_at_the_limit_is_not_reported_as_cut(tmp_path):
    (tmp_path / "exact.md").write_bytes(b"y" * 64)
    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True, max_bytes=64)
        assert done.cut == 0 and done.added == 1, done.record()
    finally:
        await engine.close()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes on this platform")
async def test_a_named_pipe_is_never_opened(tmp_path):
    """Opening a FIFO blocks until somebody writes to it, so a sync that
    reads anything that is "not a directory" hangs forever on one, and a
    hang is worse than a wrong number: nothing reports it and nothing
    recovers.

    The classification is asserted before anything is opened, so a
    regression here fails immediately instead of hanging the suite.
    """
    from scone_memory.ingestion.sync import _files

    tree(tmp_path, **{"real.md": "the real note"})
    os.mkfifo(tmp_path / "stream.md")
    reading, there, unreadable, links, special = _files(tmp_path, (".md",), 100)
    assert [path.name for path in reading] == ["real.md"], reading
    assert (there, special) == (1, 1), (there, special)

    engine = await memory()
    try:
        done = await sync_directory(engine, "default", tmp_path, apply=True)
    finally:
        await engine.close()
    assert done.files_found == 1 and done.added == 1, done.record()
    assert done.special == 1 and "not ordinary files" in done.text(), done.text()
