"""A configured sync run that schedules what it writes: the schedule is resolved
when the run is admitted and kept on the run, so every attempt writes the same
instant, a retry of the same request is the same run, and a run whose instant
has passed is not resumed."""
from __future__ import annotations

import asyncio

import pytest

from scone_memory.agents.workflow import WorkflowError
from scone_memory.core.errors import InvalidInput

from .test_directory_service import service, settled
from .test_directory_sync_forget_after import env, runner  # noqa: F401 - the clocked fixture


async def test_a_run_keeps_its_schedule_on_the_run_and_every_outcome(env):
    memory, root, _ = env
    (root / "same.txt").write_text("Unchanged source")
    host = service(env, sync=runner(env))
    try:
        await host.start("alpha", "first", collection_id="notes", forget_after="1d")
        first = await settled(host, "first")
        assert first.record.status == "completed"
        assert (first.record.spec.forget_after, first.record.spec.forget_after_asked) == ("2026-09-16T12:00:00.000Z", "1d")
        [outcome] = (await host.result("alpha", "first")).items
        assert outcome.source.status == "added" and outcome.source.forget_after == "2026-09-16T12:00:00.000Z"
        await host.start("alpha", "second", collection_id="notes", forget_after="9d")
        second = await settled(host, "second")
        [kept] = (await host.result("alpha", "second")).items
        assert kept.source.status == "unchanged" and kept.source.forget_after == "2026-09-16T12:00:00.000Z"
        assert second.record.schedule_kept == 1 and first.record.schedule_kept == 0
    finally:
        await host.aclose()
    host = service(env, sync=runner(env))
    try:
        reopened = await host.status("alpha", "first")
        assert reopened.record.spec.forget_after == "2026-09-16T12:00:00.000Z", "the schedule survives a restart"
    finally:
        await host.aclose()


async def test_a_retry_of_the_same_request_is_the_same_run_even_for_a_duration(env):
    memory, root, _ = env
    (root / "note.txt").write_text("A source")
    host = service(env, sync=runner(env))
    try:
        await host.start("alpha", "scan", collection_id="notes", forget_after="1d")
        before = await settled(host)
        memory.test_clock.now = "2026-09-15T12:05:00.000Z"
        again = await host.start("alpha", "scan", collection_id="notes", forget_after="1d")
        assert again.record == before.record, "a duration sent again names the run already admitted"
        with pytest.raises(WorkflowError, match="sync_request_conflict"):
            await host.start("alpha", "scan", collection_id="notes", forget_after="2d")
        with pytest.raises(WorkflowError, match="sync_request_conflict"):
            await host.start("alpha", "scan", collection_id="notes")
    finally:
        await host.aclose()


def test_the_run_store_takes_a_retried_duration_as_the_run_it_registered(tmp_path):
    from scone_memory.ingestion.directory_runs import DirectoryRunStore, SyncRunSpec

    registry = DirectoryRunStore(tmp_path / "runs.sqlite", key=b"d" * 32)
    try:
        spec = SyncRunSpec(collection_id="notes", configuration="a" * 64, forget_after="2026-09-16T12:00:00.000Z",
                           forget_after_asked="1d")
        first = registry.register("alpha", "scan", spec)
        later = spec.model_copy(update={"forget_after": "2026-09-16T12:05:00.000Z"})
        assert registry.register("alpha", "scan", later) == first, "a registration race on the same request is one run"
        with pytest.raises(WorkflowError, match="sync_request_conflict"):
            registry.register("alpha", "scan", later.model_copy(update={"forget_after_asked": "2d"}))
    finally:
        registry.close()


@pytest.mark.parametrize("bad", ["2020-01-01", "tomorrow", "x" * 65, 30])
async def test_a_refused_schedule_admits_no_run(env, bad):
    host = service(env, sync=runner(env))
    try:
        with pytest.raises(InvalidInput, match="forget_after"):
            await host.start("alpha", "scan", collection_id="notes", forget_after=bad)
        assert await host.request("alpha", "scan") is None
    finally:
        await host.aclose()


async def test_a_run_whose_instant_has_passed_is_not_resumed(env):
    memory, root, _ = env
    (root / "note.txt").write_text("Recoverable source")
    sync = runner(env)
    entered = asyncio.Event()
    parse = sync.parser.parse

    async def gated(*args):
        entered.set()
        await asyncio.Event().wait()

    sync.parser.parse = gated
    host = service(env, sync=sync)
    try:
        await host.start("alpha", "scan", collection_id="notes", forget_after="1h")
        await asyncio.wait_for(entered.wait(), 3)
        before = await host.status("alpha", "scan")
        await host.cancel("alpha", "scan", expected_revision=before.record.revision)
        cancelled = await settled(host)
        sync.parser.parse = parse
        memory.test_clock.now = "2026-09-15T13:00:00.000Z"
        with pytest.raises(WorkflowError, match="sync_schedule_passed"):
            await host.resume("alpha", "scan", expected_revision=cancelled.record.revision)
        memory.test_clock.now = "2026-09-15T12:59:59.000Z"
        await host.resume("alpha", "scan", expected_revision=cancelled.record.revision)
        resumed = await settled(host)
        assert resumed.record.status == "completed"
        [outcome] = (await host.result("alpha", "scan")).items
        assert outcome.source.forget_after == "2026-09-15T13:00:00.000Z", "the resumed attempt writes the admitted instant"
    finally:
        await host.aclose()


async def test_a_retry_after_an_absolute_instant_has_passed_is_still_the_same_run(env):
    memory, root, _ = env
    (root / "note.txt").write_text("A source")
    host = service(env, sync=runner(env))
    try:
        await host.start("alpha", "scan", collection_id="notes", forget_after="2026-09-15T13:00:00Z")
        before = await settled(host)
        assert before.record.status == "completed"
        memory.test_clock.now = "2026-09-15T13:30:00.000Z"
        again = await host.start("alpha", "scan", collection_id="notes", forget_after="2026-09-15T13:00:00Z")
        assert again.record == before.record, "a client retrying after a lost response gets the run it started"
        with pytest.raises(InvalidInput, match="forget_after"):
            await host.start("alpha", "other", collection_id="notes", forget_after="2026-09-15T13:00:00Z")
    finally:
        await host.aclose()


async def test_a_run_whose_walk_outlasts_its_instant_completes_writing_that_instant(env):
    memory, root, _ = env
    (root / "note.txt").write_text("A slow source")
    sync = runner(env)
    parse = sync.parser.parse

    async def slow(*args):
        memory.test_clock.now = "2026-09-15T13:30:00.000Z"
        return await parse(*args)

    sync.parser.parse = slow
    host = service(env, sync=sync)
    try:
        await host.start("alpha", "scan", collection_id="notes", forget_after="1h")
        done = await settled(host)
        assert done.record.status == "completed", done.record
        [outcome] = (await host.result("alpha", "scan")).items
        assert outcome.source.status == "added" and outcome.source.forget_after == "2026-09-15T13:00:00.000Z"
    finally:
        await host.aclose()
