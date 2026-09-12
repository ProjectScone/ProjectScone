"""Trying a parked record again, on purpose.

A record the extractor keeps failing on is parked after `max_attempts`,
which is right: a poisoned record should not burn a model call every
pass forever. But the park lives in the running process, so until now
the only way to try one again was to restart the server -- which
un-parks *everything*, including the records you had every reason to
leave alone. This is the deliberate version.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.providers.llm import ChatError
from scone_memory.ingestion.distill import DistillError, Distiller

pytestmark = pytest.mark.asyncio


class Broken:
    """A model that fails until told to stop, counting what it was asked."""

    def __init__(self) -> None:
        self.working = False
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.working:
            return ('[{"subject": "alice chen", "predicate": "works_at", '
                    '"object": "Acme Robotics"}]')
        raise ChatError("the model is down")


async def parked_engine():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    chat = Broken()
    distiller = Distiller(engine, chat, max_attempts=2, require_grounding=False)
    await engine.remember("default", "Alice Chen works at Acme Robotics.")
    for _ in range(2):
        with pytest.raises(DistillError):
            await distiller.distill_pending("default")
    return engine, chat, distiller


async def test_a_record_that_keeps_failing_is_parked():
    engine, chat, distiller = await parked_engine()
    try:
        assert len(distiller.parked("default")) == 1
        spent = chat.calls
        await distiller.distill_pending("default")
        assert chat.calls == spent, "a parked record does not spend another model call"
    finally:
        await engine.close()


async def test_retrying_a_parked_record_lets_the_next_pass_try_it():
    engine, chat, distiller = await parked_engine()
    try:
        again = await distiller.retry("default")
        assert again.unparked == 1 and again.cleared == 1, again.record()
        assert not distiller.parked("default")
        chat.working = True
        [outcome] = await distiller.distill_pending("default")
        assert outcome.error is None and outcome.added, outcome
    finally:
        await engine.close()


async def test_one_record_can_be_retried_without_unparking_the_rest():
    """The whole reason this exists: restarting the process retries
    everything, and that is exactly what an operator does not want."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        chat = Broken()
        distiller = Distiller(engine, chat, max_attempts=1, require_grounding=False)
        first = (await engine.remember("default", "Alice Chen works at Acme Robotics.")).episode_id
        second = (await engine.remember("default", "Bob Stone works at Globex.")).episode_id
        with pytest.raises(DistillError):
            await distiller.distill_pending("default")
        assert sorted(distiller.parked("default")) == sorted([first, second])
        again = await distiller.retry("default", episodes=[first])
        assert again.unparked == 1 and list(distiller.parked("default")) == [second]
    finally:
        await engine.close()


async def test_retrying_something_that_was_not_failing_says_so():
    engine, chat, distiller = await parked_engine()
    try:
        again = await distiller.retry("default", episodes=[9999])
        assert again.unparked == 0 and again.cleared == 0 and again.unknown == 1, again.record()
        assert len(distiller.parked("default")) == 1, "an unrelated id changes nothing"
        assert "1 of 1" in again.text() and "nothing recorded" in again.text(), again.text()
    finally:
        await engine.close()


async def test_a_failing_record_that_is_not_parked_yet_can_still_be_retried():
    """Clearing a record with one failure against it is not the same as
    un-parking one, and reporting them as one number would hide which
    happened."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        chat = Broken()
        distiller = Distiller(engine, chat, max_attempts=5, require_grounding=False)
        await engine.remember("default", "Alice Chen works at Acme Robotics.")
        with pytest.raises(DistillError):
            await distiller.distill_pending("default")
        assert not distiller.parked("default"), "one failure of five is not parked"
        again = await distiller.retry("default")
        assert again.cleared == 1 and again.unparked == 0, again.record()
    finally:
        await engine.close()


async def test_a_retry_of_a_bad_space_or_a_bad_id_is_refused():
    engine, chat, distiller = await parked_engine()
    try:
        with pytest.raises(InvalidInput):
            await distiller.retry("default", episodes=[0])
        with pytest.raises(InvalidInput):
            await distiller.retry("default", episodes=[])
    finally:
        await engine.close()
