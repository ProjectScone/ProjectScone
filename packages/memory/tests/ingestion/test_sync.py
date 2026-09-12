"""Keeping a space in step with a directory, including what left it.

`map` remembers files and notices when it has seen one before. What it
cannot do is notice that a file *changed* or that a file is *gone*, and
those are the two things that make the difference between an import and
a sync. The dangerous one is removal: forgetting memory because a file
is missing is destructive, so it is opt-in, previewed first, and refused
outright when the directory came back empty.
"""

from __future__ import annotations

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
