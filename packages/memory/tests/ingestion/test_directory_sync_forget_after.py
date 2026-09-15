"""A journaled directory sync that schedules what it writes.

``DirectorySync.synchronize(forget_after=...)`` resolves the schedule once, before
the journal is read, and stores that one instant on every source revision the
run writes. Each receipt says the schedule its stored revision holds: the run's
for one it wrote, the episode's own for one it left unchanged. The result names
the run's instant and counts the unchanged sources holding another. A source
whose schedule has come is forgotten, and the path is suppressed as any forget
of a managed source suppresses it: a later run does not bring it back.
"""
from __future__ import annotations

from datetime import timedelta
import itertools

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import Gone, InvalidInput
from scone_memory.core.timeutil import format_rfc3339, parse_rfc3339
from scone_memory.testing import Clock

from .test_directory_sync import TextParser, receipts

NOW = "2026-09-15T12:00:00.000Z"
START = parse_rfc3339(NOW)


@pytest.fixture(params=["memory", "sqlite"])
async def env(request, tmp_path):
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    root = tmp_path / "sources"
    root.mkdir()
    if request.param == "sqlite":
        documents, vectors = SqliteDocumentStore(tmp_path / "store.db"), SqliteVectorIndex(tmp_path / "store.db")
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    clock = Clock(NOW)
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=FileBlobStore(tmp_path / "blobs"),
                                clock=clock).open()
    engine.test_clock = clock
    yield engine, root, tmp_path
    await engine.close()


def runner(env, **changes):
    from scone_memory.ingestion.directory_sync import DirectorySync
    memory, root, tmp_path = env
    options = dict(space="alpha", journal=tmp_path / "sync.journal", key=b"k" * 32,
                   store_id="fixture-catalog", parser_revision="v1", parser=TextParser())
    return DirectorySync(memory, root, **(options | changes))


async def test_every_source_a_run_writes_carries_its_one_schedule(env):
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    (root / "notes.txt").write_text("Telescope maintenance notes")
    result = await runner(env).synchronize(forget_after="30d")
    assert result.complete and result.forget_after == "2026-10-15T12:00:00.000Z" and result.schedule_kept == 0
    for receipt in result.receipts:
        assert receipt.status == "added" and receipt.forget_after == "2026-10-15T12:00:00.000Z"
        assert (await memory.episode("alpha", receipt.episode_id)).metadata["forget_after"] == result.forget_after
    (root / "plain.txt").write_text("Unscheduled source")
    plain = await runner(env).synchronize()
    assert plain.forget_after is None and receipts(plain)["plain.txt"].forget_after is None


async def test_a_duration_is_resolved_once_for_the_whole_run(tmp_path):
    from scone_memory.ingestion.directory_sync import DirectorySync

    ticks = itertools.count()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: format_rfc3339(START + timedelta(minutes=next(ticks)))).open()
    root = tmp_path / "sources"
    root.mkdir()
    try:
        for n in range(3):
            (root / f"source{n}.txt").write_text(f"Source number {n}")
        result = await DirectorySync(engine, root, space="alpha", journal=tmp_path / "sync.journal", key=b"k" * 32,
                                     store_id="fixture-catalog", parser_revision="v1",
                                     parser=TextParser()).synchronize(forget_after="1d")
        held = {(await engine.documents.get_episode("alpha", r.episode_id)).metadata["forget_after"] for r in result.receipts}
        assert held == {result.forget_after}, f"one run, one instant: {held}"
    finally:
        await engine.close()


@pytest.mark.parametrize("bad", ["2020-01-01", NOW, "1.5d"])
async def test_a_refused_schedule_refuses_the_run_before_the_journal(env, bad):
    memory, root, tmp_path = env
    (root / "report.txt").write_text("Observatory schedule")
    with pytest.raises(InvalidInput, match="forget_after"):
        await runner(env).synchronize(forget_after=bad)
    assert not (tmp_path / "sync.journal").exists(), "nothing of the run is recorded"
    assert await memory.blobs.held("alpha") == [] and (await memory.status("alpha")).episodes == 0


async def test_an_unchanged_source_keeps_the_schedule_it_holds(env):
    memory, root, _ = env
    (root / "same.txt").write_text("Unchanged source")
    (root / "moves.txt").write_text("First revision")
    first = await runner(env).synchronize(forget_after="1d")
    (root / "moves.txt").write_text("Second revision")
    second = await runner(env).synchronize(forget_after="9d")
    same, moves = receipts(second)["same.txt"], receipts(second)["moves.txt"]
    assert same.status == "unchanged" and same.forget_after == receipts(first)["same.txt"].forget_after == "2026-09-16T12:00:00.000Z"
    assert moves.status == "updated" and moves.forget_after == "2026-09-24T12:00:00.000Z"
    assert second.forget_after == "2026-09-24T12:00:00.000Z" and second.schedule_kept == 1
    assert (await memory.episode("alpha", same.episode_id)).metadata["forget_after"] == "2026-09-16T12:00:00.000Z"
    third = await runner(env).synchronize(forget_after="2026-09-24T12:00:00Z")
    assert third.schedule_kept == 1, "only the source holding another instant is counted"
    assert (await runner(env).synchronize()).schedule_kept == 2


async def test_a_source_whose_time_has_come_is_forgotten_and_stays_suppressed(env):
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    added = receipts(await runner(env).synchronize(forget_after="1h"))["report.txt"]
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    run = await runner(env).synchronize(forget_after="1h")
    later = receipts(run)["report.txt"]
    assert later.status == "suppressed" and later.forget_after is None
    assert run.schedule_kept == 0, "only an unchanged source keeps a schedule; a suppressed one holds none"
    with pytest.raises(Gone):
        await memory.episode("alpha", added.episode_id)
    # The run reads an overdue revision as gone and leaves it to the sweep, which takes it.
    assert (await memory.forget_due("alpha")).forgotten == [added.episode_id]
    (root / "report.txt").write_text("Observatory schedule, revised")
    again = await runner(env).synchronize(forget_after="1h")
    assert receipts(again)["report.txt"].status == "suppressed" and (await memory.status("alpha")).episodes == 0


async def test_a_revision_indexed_before_an_interruption_keeps_the_schedule_it_was_stored_with(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    store = module.store_document

    async def fail_after_index(*args, **kwargs):
        await store(*args, **kwargs)
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(module, "store_document", fail_after_index)
        assert not (await runner(env).synchronize(forget_after="1d")).complete
    resumed = receipts(await runner(env).synchronize(forget_after="5d"))["report.txt"]
    assert resumed.status == "added" and resumed.forget_after == "2026-09-16T12:00:00.000Z", \
        "the replay is a duplicate of the stored revision, which keeps its schedule"


async def test_a_revision_interrupted_before_it_was_indexed_takes_the_schedule_of_the_run_that_writes_it(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")

    async def fail_before_index(*args, **kwargs):
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(module, "store_document", fail_before_index)
        assert not (await runner(env).synchronize(forget_after="1d")).complete
    resumed = receipts(await runner(env).synchronize(forget_after="5d"))["report.txt"]
    assert resumed.status == "added" and resumed.forget_after == "2026-09-20T12:00:00.000Z"
    assert (await memory.episode("alpha", resumed.episode_id)).metadata["forget_after"] == "2026-09-20T12:00:00.000Z"


async def test_a_pending_revision_of_a_file_gone_from_disk_takes_the_schedule_of_the_run_that_finishes_it(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")

    async def fail_before_index(*args, **kwargs):
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(module, "store_document", fail_before_index)
        assert not (await runner(env).synchronize(forget_after="1d")).complete
    (root / "report.txt").unlink()
    finished = receipts(await runner(env).synchronize(forget_after="3d"))["report.txt"]
    assert finished.status == "added" and finished.forget_after == "2026-09-18T12:00:00.000Z"
