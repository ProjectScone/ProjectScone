"""A directory sync that schedules what it writes.

``sync_directory(..., forget_after=...)`` resolves the schedule once, against
the engine's clock before the walk, and writes that one instant on every file
it adds or updates; the receipt names it. A file the sync does not write
(unchanged) keeps the schedule its memory holds, and the receipt counts those
whose schedule is not this run's. A refused schedule refuses the run before
anything is read or written, on a plan as on an applied sync.
"""
from __future__ import annotations

from datetime import timedelta
import io
import itertools
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import Gone, InvalidInput
from scone_memory.ingestion.sync import sync_directory
from scone_memory.runtime.cli import build_parser, run
from scone_memory.core.timeutil import format_rfc3339, parse_rfc3339
from scone_memory.testing import Clock

NOW = "2026-09-15T12:00:00.000Z"
START = parse_rfc3339(NOW)


@pytest.fixture
async def memory():
    clock = Clock(NOW)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(), clock=clock).open()
    engine.test_clock = clock
    yield engine
    await engine.close()


def tree(root, **files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


async def schedules(engine, marker) -> dict[str, str | None]:
    return {e.source: e.metadata.get("forget_after") for e in await engine.episodes("default", {"sync": marker})}


async def test_every_file_a_sync_writes_carries_its_one_schedule(memory, tmp_path):
    tree(tmp_path, **{"a.py": "def a(): pass", "notes/b.md": "# harbour notes", "c.txt": "closed in November"})
    done = await sync_directory(memory, "default", tmp_path, marker="repo", apply=True, forget_after="30d")
    assert done.added == 3 and done.forget_after == "2026-10-15T12:00:00.000Z"
    assert await schedules(memory, "repo") == {"a.py": done.forget_after, "notes/b.md": done.forget_after,
                                               "c.txt": done.forget_after}
    assert done.record()["forget_after"] == "2026-10-15T12:00:00.000Z" and "schedule_kept" not in done.record()
    assert "every file written is to be forgotten after 2026-10-15T12:00:00.000Z" in done.text()
    plain = await sync_directory(memory, "default", tree(tmp_path / "other", **{"d.md": "# other"}), marker="other", apply=True)
    assert plain.forget_after is None and "forget_after" not in plain.record()
    assert "forgotten after" not in plain.text()


async def test_a_duration_is_resolved_once_for_the_whole_run(tmp_path):
    ticks = itertools.count()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: format_rfc3339(START + timedelta(minutes=next(ticks)))).open()
    try:
        tree(tmp_path, **{f"f{n}.md": f"# file number {n}" for n in range(4)})
        done = await sync_directory(engine, "default", tmp_path, marker="repo", apply=True, forget_after="1h")
        held = set((await schedules(engine, "repo")).values())
        assert held == {done.forget_after}, f"one run, one instant: {held}"
    finally:
        await engine.close()


async def test_a_walk_that_outlasts_its_schedule_writes_every_file_with_the_one_instant(tmp_path):
    ticks = itertools.count()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: format_rfc3339(START + timedelta(minutes=next(ticks)))).open()
    try:
        tree(tmp_path, **{f"f{n}.md": f"# file number {n}" for n in range(8)})
        done = await sync_directory(engine, "default", tmp_path, marker="repo", apply=True, forget_after="3m")
        assert done.added == 8 and done.forget_after == "2026-09-15T12:03:00.000Z"
        assert (await engine.status("default")).episodes == 8, "the run is not refused part way through its walk"
        report = await engine.forget_due("default")
        assert len(report.forgotten) == 8 and {item.forget_after for item in report.items} == {done.forget_after}
    finally:
        await engine.close()


async def test_a_plan_names_the_schedule_it_would_write_and_writes_nothing(memory, tmp_path):
    tree(tmp_path, **{"a.md": "# plan"})
    plan = await sync_directory(memory, "default", tmp_path, marker="repo", forget_after="2026-10-01")
    assert not plan.applied and plan.added == 1 and plan.forget_after == "2026-10-01T00:00:00.000Z"
    assert "would sync" in plan.text() and "every file it would write is to be forgotten after 2026-10-01T00:00:00.000Z" in plan.text()
    assert (await memory.status("default")).episodes == 0


@pytest.mark.parametrize("apply", [True, False])
@pytest.mark.parametrize("bad", ["2020-01-01", NOW, "tomorrow"])
async def test_a_refused_schedule_refuses_the_run_before_it_writes(memory, tmp_path, apply, bad):
    tree(tmp_path, **{"a.md": "# refused"})
    with pytest.raises(InvalidInput, match="forget_after"):
        await sync_directory(memory, "default", tmp_path, marker="repo", apply=apply, forget_after=bad)
    assert (await memory.status("default")).episodes == 0
    assert await memory.events.query("default", kind="sync.completed") == []


async def test_an_unchanged_file_keeps_the_schedule_it_holds_and_the_receipt_counts_it(memory, tmp_path):
    tree(tmp_path, **{"same.md": "# stays the same", "moves.md": "# first words", "also.md": "# also scheduled"})
    first = await sync_directory(memory, "default", tmp_path, marker="repo", apply=True, forget_after="1d")
    assert first.schedule_kept == 0
    tree(tmp_path, **{"moves.md": "# second words"})
    second = await sync_directory(memory, "default", tmp_path, marker="repo", apply=True, forget_after="9d")
    assert (second.updated, second.unchanged) == (1, 2)
    assert second.schedule_kept == 2, "both unchanged files hold another schedule, which this run did not write"
    assert second.record()["schedule_kept"] == 2
    assert "2 unchanged file(s) keep the schedule their memory holds" in second.text()
    assert await schedules(memory, "repo") == {"same.md": "2026-09-16T12:00:00.000Z",
                                               "also.md": "2026-09-16T12:00:00.000Z",
                                               "moves.md": "2026-09-24T12:00:00.000Z"}
    again = await sync_directory(memory, "default", tmp_path, marker="repo", apply=True, forget_after="2026-09-16T12:00:00Z")
    assert again.unchanged == 3 and again.schedule_kept == 1, "only the file holding another instant is counted"
    unscheduled = await sync_directory(memory, "default", tmp_path, marker="repo", apply=True)
    assert unscheduled.schedule_kept == 3 and "schedule_kept" in unscheduled.record()


async def test_a_file_whose_memory_is_past_its_time_is_written_afresh(memory, tmp_path):
    tree(tmp_path, **{"a.md": "# kept for an hour"})
    await sync_directory(memory, "default", tmp_path, marker="repo", apply=True, forget_after="1h")
    [before] = await memory.episodes("default", {"sync": "repo"})
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    done = await sync_directory(memory, "default", tmp_path, marker="repo", apply=True, forget_after="1h")
    assert (done.added, done.unchanged, done.schedule_kept) == (1, 0, 0)
    with pytest.raises(Gone):
        await memory.episode("default", before.episode_id)
    assert await schedules(memory, "repo") == {"a.md": "2026-09-15T15:00:00.000Z"}


async def test_the_sync_command_takes_a_schedule(memory, tmp_path):
    tree(tmp_path, **{"a.md": "# from the command line"})
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "sync", str(tmp_path), "--apply", "--marker", "repo",
                                                "--forget-after", "2w"]), memory, io.StringIO(""), out)
    assert code == 0 and json.loads(out.getvalue())["forget_after"] == "2026-09-29T12:00:00.000Z"
    assert await schedules(memory, "repo") == {"a.md": "2026-09-29T12:00:00.000Z"}
    with pytest.raises(InvalidInput, match="forget_after"):
        await run(build_parser().parse_args(["sync", str(tmp_path), "--forget-after", "5y"]), memory, io.StringIO(""), out)
