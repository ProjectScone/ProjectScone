"""A journaled directory sync that schedules what it writes.

``DirectorySync.synchronize(forget_after=...)`` resolves the schedule once, before
the journal is read, and stores that one instant on every source revision the
run writes. Each receipt says the schedule its stored revision holds: the run's
for one it wrote, the episode's own for one it left unchanged. The result names
the run's instant and counts the unchanged sources holding another. A source
whose schedule has come is imported afresh by the next run that sees the file,
as ``sync`` does; one forgotten by hand before its time stays suppressed. A
pending revision keeps the schedule it was prepared under, whichever run
finishes it.
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


async def test_a_source_whose_time_has_come_is_imported_afresh_by_the_next_run(env):
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    added = receipts(await runner(env).synchronize(forget_after="1h"))["report.txt"]
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    run = await runner(env).synchronize(forget_after="1h")
    again = receipts(run)["report.txt"]
    assert again.status == "updated" and again.episode_id != added.episode_id
    assert again.forget_after == "2026-09-15T15:00:00.000Z" and run.schedule_kept == 0
    with pytest.raises(Gone):
        await memory.episode("alpha", added.episode_id)
    assert (await memory.forget_due("alpha")).forgotten == [], "the run forgot the overdue revision it replaced"
    memory.test_clock.now = "2026-09-15T16:00:00.000Z"
    assert (await memory.forget_due("alpha")).forgotten == [again.episode_id]
    unchanged = receipts(await runner(env).synchronize())["report.txt"]
    assert unchanged.status == "updated" and unchanged.forget_after is None, "a swept source is not suppressed"
    (root / "report.txt").write_text("Observatory schedule, revised")
    revised = receipts(await runner(env).synchronize(forget_after="1h"))["report.txt"]
    assert revised.status == "updated" and (await memory.status("alpha")).episodes == 1


async def test_a_source_forgotten_by_hand_before_its_time_stays_suppressed(env):
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    added = receipts(await runner(env).synchronize(forget_after="1d"))["report.txt"]
    await memory.forget("alpha", added.episode_id)
    memory.test_clock.now = "2026-09-17T12:00:00.000Z"
    (root / "report.txt").write_text("Observatory schedule, revised")
    later = receipts(await runner(env).synchronize(forget_after="1d"))["report.txt"]
    assert later.status == "suppressed" and (await memory.status("alpha")).episodes == 0


async def test_a_walk_that_outlasts_its_schedule_writes_every_source_with_the_one_instant(tmp_path):
    from scone_memory.ingestion.directory_sync import DirectorySync

    ticks = itertools.count()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: format_rfc3339(START + timedelta(minutes=next(ticks)))).open()
    root = tmp_path / "sources"
    root.mkdir()
    try:
        for n in range(4):
            (root / f"source{n}.py").write_text(f"import module{n}\n")
        result = await DirectorySync(engine, root, space="alpha", journal=tmp_path / "sync.journal", key=b"k" * 32,
                                     store_id="fixture-catalog", parser_revision="v1").synchronize(forget_after="2m")
        assert result.complete, [(r.path, r.status, r.code) for r in result.receipts]
        assert {r.status for r in result.receipts} == {"added"} and result.claims == 4, "claims are recorded after the instant too"
        assert result.forget_after == "2026-09-15T12:02:00.000Z"
        assert {r.forget_after for r in result.receipts} == {result.forget_after}
        assert len((await engine.forget_due("alpha")).forgotten) == 4
    finally:
        await engine.close()


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


async def test_a_revision_interrupted_before_it_was_indexed_keeps_the_schedule_it_was_prepared_under(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")

    async def fail_before_index(*args, **kwargs):
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(module, "store_document", fail_before_index)
        assert not (await runner(env).synchronize(forget_after="1d")).complete
    resumed = receipts(await runner(env).synchronize())["report.txt"]
    assert resumed.status == "added" and resumed.forget_after == "2026-09-16T12:00:00.000Z", \
        "a run asking for no schedule does not drop the one the revision was prepared under"
    assert (await memory.episode("alpha", resumed.episode_id)).metadata["forget_after"] == "2026-09-16T12:00:00.000Z"
    memory.test_clock.now = "2026-09-17T12:00:00.000Z"
    assert (await memory.forget_due("alpha")).forgotten == [resumed.episode_id]


async def test_a_revision_finished_after_the_time_it_was_prepared_for_is_imported_afresh(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")

    async def fail_before_index(*args, **kwargs):
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(module, "store_document", fail_before_index)
        assert not (await runner(env).synchronize(forget_after="1h")).complete
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    run = await runner(env).synchronize()
    resumed = receipts(run)["report.txt"]
    assert run.complete and resumed.status == "updated" and resumed.forget_after is None
    assert (await memory.status("alpha")).episodes == 1 and (await memory.forget_due("alpha")).forgotten == []


async def test_a_pending_revision_of_a_file_gone_from_disk_keeps_the_schedule_it_was_prepared_under(env, monkeypatch):
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
    assert finished.status == "added" and finished.forget_after == "2026-09-16T12:00:00.000Z"
    assert (await memory.episode("alpha", finished.episode_id)).metadata["forget_after"] == finished.forget_after


def interrupt(patch, module, *, after_index: bool):
    store = module.store_document

    async def stopped(*args, **kwargs):
        if after_index:
            await store(*args, **kwargs)
        raise OSError("interrupted")

    patch.setattr(module, "store_document", stopped)


async def test_a_replacement_whose_old_revision_was_swept_meanwhile_finishes(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    old = receipts(await runner(env).synchronize(forget_after="1h"))["report.txt"]
    (root / "report.txt").write_text("Observatory schedule, revised")
    with monkeypatch.context() as patch:
        interrupt(patch, module, after_index=False)
        assert not (await runner(env).synchronize(forget_after="1d")).complete
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    assert (await memory.forget_due("alpha")).forgotten == [old.episode_id]
    finished = receipts(await runner(env).synchronize())["report.txt"]
    assert finished.status == "updated" and finished.forget_after == "2026-09-16T12:00:00.000Z", \
        "an old revision taken at its time does not suppress the replacement"


@pytest.mark.parametrize("after_retire", [False, True])
async def test_a_pending_revision_swept_before_its_replacement_finished_is_read_afresh(env, monkeypatch, after_retire):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    if after_retire:
        assert receipts(await runner(env).synchronize())["report.txt"].status == "added"
        (root / "report.txt").write_text("Observatory schedule, revised")
    with monkeypatch.context() as patch:
        if after_retire:
            async def refuse(*args, **kwargs):
                raise OSError("interrupted")

            # Stopped after the retire intent is saved, before the old revision goes.
            patch.setattr(module.DirectorySync, "_close_claims", refuse)
        else:
            # Stopped after the store, before the journal records the episode.
            interrupt(patch, module, after_index=True)
        assert not (await runner(env).synchronize(forget_after="1h")).complete
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    assert len((await memory.forget_due("alpha")).forgotten) == 1, "only the scheduled pending revision is due"
    run = await runner(env).synchronize(forget_after="1d")
    again = receipts(run)["report.txt"]
    assert run.complete and again.status == "added" and again.forget_after == "2026-09-16T14:00:00.000Z", \
        [(r.path, r.status, r.code) for r in run.receipts]
    assert (await memory.episode("alpha", again.episode_id)).metadata["forget_after"] == again.forget_after
    assert (await memory.status("alpha")).episodes == 1, "a previous revision still held goes with the swept one"


async def test_a_missing_file_whose_revision_was_swept_is_absent_and_may_return(env):
    memory, root, _ = env
    (root / "report.txt").write_text("Observatory schedule")
    receipts(await runner(env).synchronize(forget_after="1h"))
    (root / "report.txt").unlink()
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    assert len((await memory.forget_due("alpha")).forgotten) == 1
    gone = receipts(await runner(env).synchronize(delete_missing=True))["report.txt"]
    assert gone.status == "absent"
    (root / "report.txt").write_text("Observatory schedule, back again")
    assert receipts(await runner(env).synchronize())["report.txt"].status == "added"


async def test_claims_of_a_replacement_swept_before_it_finished_are_closed_and_stated_again(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    from scone_memory.ingestion.directory_sync import DirectorySync

    from .test_directory_sync import MODULE, SHORTER

    memory, root, tmp_path = env
    options = dict(space="alpha", journal=tmp_path / "claims.journal", key=b"k" * 32, store_id="fixture-catalog",
                   parser_revision="v1")
    (root / "store.py").write_text(MODULE)
    old = receipts(await DirectorySync(memory, root, **options).synchronize())["store.py"]
    (root / "store.py").write_text("import sys\n" + SHORTER)
    with monkeypatch.context() as patch:
        async def refuse(*args, **kwargs):
            raise OSError("interrupted")

        patch.setattr(module.DirectorySync, "_close_claims", refuse)
        assert not (await DirectorySync(memory, root, **options).synchronize(forget_after="1h")).complete
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    [swept] = (await memory.forget_due("alpha")).forgotten
    again = receipts(await DirectorySync(memory, root, **options).synchronize())["store.py"]
    assert again.status == "added" and again.claims >= 4

    async def active(episode_id):
        return {(f.predicate, f.object) for f in await memory.documents.facts_for_graph("alpha", episode_id, 500)
                if f.status == "active"}

    stated = await active(old.episode_id) | await active(swept) | await active(again.episode_id)
    assert ("imports", "sys") in stated and ("defines", "store.py:Store.read") in stated
    assert ("imports", "os") not in stated and ("defines", "store.py:Store.write") not in stated, \
        "what the file no longer says is closed, though neither revision it was cited to finished"
    (root / "store.py").write_text("import json\n")
    receipts(await DirectorySync(memory, root, **options).synchronize())
    left = await active(old.episode_id) | await active(swept) | await active(again.episode_id)
    assert left <= {("imports", "json")}, left


@pytest.mark.parametrize("swept", [False, True])
async def test_a_source_read_afresh_after_its_time_closes_what_the_file_no_longer_says(env, swept):
    from scone_memory.ingestion.directory_sync import DirectorySync

    from .test_directory_sync import MODULE, SHORTER

    memory, root, tmp_path = env
    options = dict(space="alpha", journal=tmp_path / "claims.journal", key=b"k" * 32, store_id="fixture-catalog",
                   parser_revision="v1")
    (root / "store.py").write_text(MODULE)
    old = receipts(await DirectorySync(memory, root, **options).synchronize(forget_after="1h"))["store.py"]
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    if swept:
        assert (await memory.forget_due("alpha")).forgotten == [old.episode_id]
    (root / "store.py").write_text(SHORTER)
    new = receipts(await DirectorySync(memory, root, **options).synchronize())["store.py"]
    assert new.status == "updated" and new.claims_closed is not None and new.claims_closed >= 2
    facts = await memory.documents.facts_for_graph("alpha", old.episode_id, 500)
    active = {(f.predicate, f.object) for f in facts if f.status == "active"}
    assert ("imports", "os") not in active and ("defines", "store.py:Store.write") not in active
    assert ("defines", "store.py:Store.read") in active, "what the file still says stands"


async def test_a_missing_file_taken_at_its_time_has_its_claims_closed(env):
    from scone_memory.ingestion.directory_sync import DirectorySync

    from .test_directory_sync import MODULE

    memory, root, tmp_path = env
    options = dict(space="alpha", journal=tmp_path / "claims.journal", key=b"k" * 32, store_id="fixture-catalog",
                   parser_revision="v1")
    (root / "store.py").write_text(MODULE)
    old = receipts(await DirectorySync(memory, root, **options).synchronize(forget_after="1h"))["store.py"]
    (root / "store.py").unlink()
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    assert (await memory.forget_due("alpha")).forgotten == [old.episode_id]
    gone = receipts(await DirectorySync(memory, root, **options).synchronize(delete_missing=True))["store.py"]
    assert gone.status == "absent" and gone.claims_closed is not None and gone.claims_closed >= 4
    assert all(f.status != "active" for f in await memory.documents.facts_for_graph("alpha", old.episode_id, 500))
