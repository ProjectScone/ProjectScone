"""Crash recovery: a remember interrupted between its steps is completed
or forgotten on the next open, never left half-present. Runs over every
backend pair through the shared ``engine`` fixture. A crash is played by
raising something the batch's rollback does not catch (a BaseException
that is not an Exception), which leaves the store exactly as a killed
process would."""

from __future__ import annotations

import pytest

from ..engine import MemoryEngine
from ..models import RecallItem


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so the batch's
    own rollback cannot see it."""


async def _reopen(engine: MemoryEngine) -> MemoryEngine:
    """A fresh engine over the same stores, as a restart would give."""
    fresh = MemoryEngine(
        engine.documents, engine.vectors, engine.embedder, chunk_target=engine.chunk_target, clock=engine.clock, events=engine.events
    )
    return await fresh.open()


def _vector_lane(items: list[RecallItem]) -> bool:
    return any("vector" in i.lanes for i in items)


async def test_a_write_cut_off_before_its_vectors_is_completed_on_open(engine, monkeypatch):
    real_upsert = engine.vectors.upsert

    async def die(*a, **kw):
        raise Crash()

    monkeypatch.setattr(engine.vectors, "upsert", die)
    with pytest.raises(Crash):
        await engine.remember("default", "the deploy runbook lives in the ops wiki", tags=["ops"], metadata={"user_id": "mark"})
    monkeypatch.setattr(engine.vectors, "upsert", real_upsert)
    assert await engine.documents.inflight() != [], "the mark outlives the crash"
    assert (await engine.status("default")).episodes == 1, "the rows landed; only the vectors are missing"
    before = await engine.recall("default", "deploy runbook ops wiki")
    assert before.items and not _vector_lane(before.items), "found by the lexical lane only, as the half-written state implies"

    reopened = await _reopen(engine)
    report_marks = await engine.documents.inflight()
    assert report_marks == [], "recover() cleared the mark"
    after = await reopened.recall("default", "deploy runbook ops wiki")
    assert [i.episode_id for i in after.items] == [1] and _vector_lane(after.items), "the vectors exist now"
    assert after.items[0].tags == ("ops",) and after.items[0].metadata == {"user_id": "mark"}
    [event] = await reopened.events.query("default", kind="recover")
    assert event.payload == {"marks": 1, "completed": 1, "rechunked": 0, "forgotten": 0}


async def test_a_write_cut_off_before_its_chunks_is_rechunked_and_embedded(engine, monkeypatch):
    real_insert = engine.documents.insert_chunks

    async def die(*a, **kw):
        raise Crash()

    monkeypatch.setattr(engine.documents, "insert_chunks", die)
    with pytest.raises(Crash):
        await engine.remember("default", "a long note " * 60, created_at="2024-01-01")
    monkeypatch.setattr(engine.documents, "insert_chunks", real_insert)
    assert (await engine.status("default")).episodes == 1 and (await engine.status("default")).chunks == 0

    reopened = await _reopen(engine)
    status = await reopened.status("default")
    assert status.episodes == 1 and status.chunks > 0, "chunks were rebuilt from the stored content"
    chunks = await reopened.documents.chunks_of("default", 1)
    assert [c.ordinal for c in chunks] == list(range(len(chunks))) and "".join(c.text for c in chunks) == "a long note " * 60
    after = await reopened.recall("default", "a long note")
    assert after.items and _vector_lane(after.items)
    [event] = await reopened.events.query("default", kind="recover")
    assert event.payload["rechunked"] == 1 and event.payload["completed"] == 1


async def test_a_mark_with_nothing_behind_it_is_forgotten(engine, monkeypatch):
    async def die(*a, **kw):
        raise Crash()

    monkeypatch.setattr(engine.documents, "insert_episode", die)
    with pytest.raises(Crash):
        await engine.remember("default", "never landed")
    monkeypatch.undo()
    assert len(await engine.documents.inflight()) == 1
    reopened = await _reopen(engine)
    assert await engine.documents.inflight() == []
    assert (await reopened.status("default")).episodes == 0
    [event] = await reopened.events.query("default", kind="recover")
    assert event.payload == {"marks": 1, "completed": 0, "rechunked": 0, "forgotten": 1}


async def test_a_completed_write_leaves_no_mark_and_a_clean_open_records_nothing(engine):
    await engine.remember("default", "all the way through")
    assert await engine.documents.inflight() == []
    reopened = await _reopen(engine)
    assert await reopened.events.query("default", kind="recover") == []
    report = await reopened.recover()
    assert (report.completed, report.rechunked, report.forgotten) == (0, 0, 0)


async def test_a_failed_batch_that_the_engine_rolls_back_leaves_no_mark(engine, monkeypatch):
    async def refuse(*a, **kw):
        raise RuntimeError("index offline")

    monkeypatch.setattr(engine.vectors, "upsert", refuse)
    with pytest.raises(RuntimeError):
        await engine.remember("default", "rolled back")
    monkeypatch.undo()
    assert await engine.documents.inflight() == [], "an orderly failure cleans its own mark"
    assert (await engine.status("default")).episodes == 0


__all__ = [
    "test_a_write_cut_off_before_its_vectors_is_completed_on_open",
    "test_a_write_cut_off_before_its_chunks_is_rechunked_and_embedded",
    "test_a_mark_with_nothing_behind_it_is_forgotten",
    "test_a_completed_write_leaves_no_mark_and_a_clean_open_records_nothing",
    "test_a_failed_batch_that_the_engine_rolls_back_leaves_no_mark",
]
