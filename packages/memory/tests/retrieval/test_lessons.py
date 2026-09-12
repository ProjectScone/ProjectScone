"""What a search taught us, and whether it still applies.

A framework that never learns from being wrong repeats itself. Recording
how a recall turned out is cheap, and the accumulated judgements are
worth reading -- but only with the one thing that makes them safe: a
lesson about evidence that has since **changed** must say so rather than
be quietly applied. A confident lesson about text that no longer exists
is worse than no lesson.

Three outcomes rather than a boolean, because "this was a dead end" and
"this was wrong, here is the right answer" carry different information
and rolling them together loses the second.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog,
                          InMemoryVectorIndex, MemoryEngine)
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.lessons import MAX_LESSONS, lessons, record_outcome

pytestmark = pytest.mark.asyncio


async def memory():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              events=InMemoryEventLog()).open()


async def searched(engine, text="The harbour crane was repainted in May.", query="crane repainted"):
    said = await engine.remember("default", text, source="crane.txt")
    found = await engine.recall("default", query, limit=3)
    assert found.event_id is not None and found.items, "the search has to be recorded"
    return said, found


async def test_the_three_outcomes_are_counted_apart():
    """"dead end" and "corrected" are different facts. One count for both
    would lose the correction, which is the half that says what to do."""
    engine = await memory()
    try:
        _, found = await searched(engine)
        chunk = found.items[0].chunk_id
        await record_outcome(engine, "default", found.event_id, chunk, "useful")
        _, again = await searched(engine)
        await record_outcome(engine, "default", again.event_id, again.items[0].chunk_id,
                             "corrected", note="it was June, not May")
        learned = await lessons(engine, "default")
        [one] = learned.lessons
        assert (one.useful, one.dead_end, one.corrected) == (1, 0, 1), one
        assert one.corrections == ("it was June, not May",), one.corrections
    finally:
        await engine.close()


async def test_a_lesson_about_evidence_that_changed_says_so():
    """The rule that makes the rest safe. The judgement was about text
    that has since been replaced, so applying it silently would carry a
    verdict onto evidence nobody judged."""
    from scone_memory.ingestion.records import Record

    engine = await memory()
    try:
        said, found = await searched(engine)
        await record_outcome(engine, "default", found.event_id, found.items[0].chunk_id, "useful")
        await engine.replace("default", Record(content="The crane was repainted in June.",
                                               kind="note", source="crane.txt",
                                               dedup_key="crane.txt"))
        learned = await lessons(engine, "default")
        [one] = learned.lessons
        assert one.evidence == "changed", one
        assert "re-verify" in one.why, one.why
    finally:
        await engine.close()


async def test_a_lesson_about_evidence_that_is_gone_is_not_reported_as_fresh():
    engine = await memory()
    try:
        said, found = await searched(engine)
        await record_outcome(engine, "default", found.event_id, found.items[0].chunk_id, "useful")
        await engine.forget("default", said.episode_id)
        learned = await lessons(engine, "default")
        [one] = learned.lessons
        assert one.evidence == "gone", one
        assert "no longer there" in one.why, one.why
    finally:
        await engine.close()


async def test_a_judgement_with_no_recorded_hash_says_it_cannot_tell():
    """An older outcome event carries no hash, so its freshness is
    unknown -- which must not be reported as fresh. "I did not check" is
    not "I checked and it is fine"."""
    engine = await memory()
    try:
        said, found = await searched(engine)
        await engine._emit("default", "recall.outcome", {
            "recall_event_id": found.event_id, "chunk_id": found.items[0].chunk_id,
            "episode_id": said.episode_id, "outcome": "useful", "note": None})
        learned = await lessons(engine, "default")
        [one] = learned.lessons
        assert one.evidence == "unknown", one
        assert "cannot tell" in one.why, one.why
    finally:
        await engine.close()


async def test_an_unchanged_source_is_fresh():
    engine = await memory()
    try:
        _, found = await searched(engine)
        await record_outcome(engine, "default", found.event_id, found.items[0].chunk_id, "useful")
        learned = await lessons(engine, "default")
        [one] = learned.lessons
        assert one.evidence == "fresh" and one.useful == 1, one
    finally:
        await engine.close()


async def test_an_outcome_nobody_returned_is_refused():
    engine = await memory()
    try:
        _, found = await searched(engine)
        with pytest.raises(InvalidInput):
            await record_outcome(engine, "default", found.event_id, 999_999, "useful")
        with pytest.raises(InvalidInput):
            await record_outcome(engine, "default", found.event_id, found.items[0].chunk_id,
                                 "delighted")
    finally:
        await engine.close()


async def test_more_lessons_than_are_returned_are_reported_not_hidden():
    engine = await memory()
    try:
        for n in range(3):
            said = await engine.remember("default", f"Note number {n} about the crane.",
                                         source=f"n{n}.txt")
            found = await engine.recall("default", f"note number {n} crane", limit=3)
            hit = next((i for i in found.items if i.episode_id == said.episode_id), None)
            assert hit is not None, "the note has to come back to be judged"
            await record_outcome(engine, "default", found.event_id, hit.chunk_id, "useful")
        learned = await lessons(engine, "default", limit=2)
        assert len(learned.lessons) == 2 and learned.found == 3 and learned.capped
        assert "3" in learned.text() and "2" in learned.text(), learned.text()
    finally:
        await engine.close()
