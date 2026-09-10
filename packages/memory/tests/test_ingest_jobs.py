"""Durable ingest jobs: what a batch became, and how far it has got.

Two receipts, never one. Chunks and vectors land together, so a record
is searchable the moment its batch lands; claims are settled later by a
model, and may never be settled at all. Reporting one number for both
would tell a person their knowledge is ready when it is only findable.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput, NotFound
from scone_memory.memory.engine import Record


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


def records(*texts):
    return [Record(content=text) for text in texts]


async def test_a_batch_becomes_a_job_with_one_receipt_per_record(engine):
    job = await engine.ingest_batch("alpha", records("the first note", "the second note"))
    assert job.job_id and job.space == "alpha" and job.created_at
    assert [i.index for i in job.items] == [0, 1]
    assert all(i.episode_id > 0 and i.searchable_at and i.outcome == "accepted" for i in job.items)
    assert job.searchable == 2 and job.consolidated == 0
    assert job.state == "searchable", "the words are findable; nothing has been read out of them yet"
    found = await engine.recall("alpha", "second note")
    assert any("second" in item.text for item in found.items)


async def test_the_two_receipts_move_apart(engine):
    job = await engine.ingest_batch("alpha", records("Ana moved to Lisbon", "Carol drinks black coffee"))
    assert (job.searchable, job.consolidated) == (2, 0)

    await engine.note_consolidated("alpha", [job.items[0].episode_id])
    again = await engine.job("alpha", job.job_id)
    assert (again.searchable, again.consolidated) == (2, 1), "one has been read, the other has not"
    assert again.items[0].consolidated_at and again.items[1].consolidated_at is None
    assert again.state == "searchable", "a job is only done when every record is"

    await engine.note_consolidated("alpha", [job.items[1].episode_id])
    assert (await engine.job("alpha", job.job_id)).state == "consolidated"


async def test_a_job_outlives_the_process_that_made_it(tmp_path):
    path = tmp_path / "memory.db"
    first = await MemoryEngine(SqliteDocumentStore(path), InMemoryVectorIndex(), HashEmbedder()).open()
    job = await first.ingest_batch("alpha", records("something worth keeping"))
    await first.note_consolidated("alpha", [job.items[0].episode_id])
    await first.documents.close()

    second = await MemoryEngine(SqliteDocumentStore(path), InMemoryVectorIndex(), HashEmbedder()).open()
    same = await second.job("alpha", job.job_id)
    assert same.items[0].episode_id == job.items[0].episode_id
    assert same.searchable == 1 and same.consolidated == 1, "both receipts survived the restart"
    await second.documents.close()


async def test_cancelling_stops_what_is_not_settled_and_leaves_what_is(engine):
    job = await engine.ingest_batch("alpha", records("one", "two"))
    await engine.note_consolidated("alpha", [job.items[0].episode_id])

    cancelled = await engine.cancel_job("alpha", job.job_id)
    assert cancelled.state == "cancelled" and cancelled.cancelled_at
    assert cancelled.items[0].consolidated_at and cancelled.items[0].state == "consolidated", \
        "what was already read stays read, and still says so"
    assert cancelled.items[1].state == "cancelled" and cancelled.items[1].searchable_at, \
        "and what was only searchable stays searchable: cancelling is not deleting"
    assert (await engine.recall("alpha", "two")).items, "the record itself is untouched"

    with pytest.raises(InvalidInput, match="cancelled"):
        await engine.cancel_job("alpha", job.job_id)


async def test_the_same_request_twice_is_one_job(engine):
    first = await engine.ingest_batch("alpha", records("only once"), request_id="req-1")
    second = await engine.ingest_batch("alpha", records("only once"), request_id="req-1")
    assert second.job_id == first.job_id, "a retried request is the same job, not a second one"
    assert len((await engine.jobs("alpha"))) == 1
    assert [i.episode_id for i in second.items] == [i.episode_id for i in first.items]


async def test_jobs_are_listed_newest_first_and_stay_in_their_space(engine):
    older = await engine.ingest_batch("alpha", records("older"))
    newer = await engine.ingest_batch("alpha", records("newer"))
    await engine.ingest_batch("beta", records("someone else's"))

    listed = await engine.jobs("alpha")
    assert [j.job_id for j in listed] == [newer.job_id, older.job_id]
    assert all(j.space == "alpha" for j in listed)
    with pytest.raises(NotFound):
        await engine.job("beta", older.job_id)


async def test_an_unknown_job_is_not_found(engine):
    with pytest.raises(NotFound, match="job"):
        await engine.job("alpha", "no-such-job")


async def test_the_extractor_sets_the_second_receipt(engine):
    """Nothing else may set it: a record is consolidated when a model has
    read it, and a record it could get nothing out of is finished too."""
    import json

    from scone_memory import FakeChat
    from scone_memory.ingestion.distill import Distiller

    job = await engine.ingest_batch("alpha", records("Ana moved to Lisbon.", "A line with nothing in it."))
    reply = json.dumps([{"subject": "Ana", "predicate": "moved_to", "object": "Lisbon", "confidence": 0.9,
                         "statement_type": "observation", "quote": "Ana moved to Lisbon."}])
    await Distiller(engine, FakeChat([reply, "[]"])).distill_pending("alpha")

    done = await engine.job("alpha", job.job_id)
    assert done.consolidated == 2 and done.state == "consolidated", "both were read; one of them said nothing"
    assert all(item.consolidated_at for item in done.items)


async def test_a_store_that_cannot_keep_jobs_says_so(engine, monkeypatch):
    monkeypatch.setattr(engine.documents, "create_job", None)
    with pytest.raises(InvalidInput, match="does not record ingest jobs"):
        await engine.ingest_batch("alpha", records("nowhere to put the receipt"))


async def test_a_failing_extractor_is_recorded_against_the_record_it_failed_on(engine):
    """G07 wants per-stage errors and retries, not one flag for the batch:
    which record could not be read, what went wrong, and how many attempts
    it has had."""
    job = await engine.ingest_batch("alpha", records("a line the model chokes on", "a fine line"))
    first, second = job.items[0].episode_id, job.items[1].episode_id

    await engine.note_failed("alpha", first, "ChatError: upstream said 503")
    marked = await engine.job("alpha", job.job_id)
    assert marked.items[0].state == "failed" and marked.items[0].attempts == 1
    assert "503" in marked.items[0].error and marked.items[0].searchable_at, "it is still findable, just not read"
    assert marked.items[1].state == "searchable" and marked.items[1].error is None
    assert marked.state == "failed", "a batch with a record nobody could read is not simply searchable"

    await engine.note_failed("alpha", first, "ChatError: upstream said 503 again")
    assert (await engine.job("alpha", job.job_id)).items[0].attempts == 2, "attempts count the tries, not the errors seen"


async def test_a_retry_that_works_clears_the_failure_and_keeps_the_count(engine):
    job = await engine.ingest_batch("alpha", records("the awkward one"))
    episode = job.items[0].episode_id
    await engine.note_failed("alpha", episode, "ChatError: timed out")
    await engine.note_consolidated("alpha", [episode])

    healed = (await engine.job("alpha", job.job_id)).items[0]
    assert healed.state == "consolidated" and healed.consolidated_at and healed.error is None
    assert healed.attempts == 1, "the record of having had to retry stays"
    assert (await engine.job("alpha", job.job_id)).state == "consolidated"


async def test_a_failure_outlives_the_process(tmp_path):
    path = tmp_path / "memory.db"
    first = await MemoryEngine(SqliteDocumentStore(path), InMemoryVectorIndex(), HashEmbedder()).open()
    job = await first.ingest_batch("alpha", records("one that will fail"))
    await first.note_failed("alpha", job.items[0].episode_id, "ChatError: no model configured")
    await first.documents.close()

    second = await MemoryEngine(SqliteDocumentStore(path), InMemoryVectorIndex(), HashEmbedder()).open()
    same = (await second.job("alpha", job.job_id)).items[0]
    assert same.state == "failed" and same.attempts == 1 and "no model" in same.error

    # A retry in the new process clears the error and keeps the count.
    await second.note_consolidated("alpha", [same.episode_id])
    healed = (await second.job("alpha", job.job_id)).items[0]
    assert healed.state == "consolidated" and healed.error is None and healed.attempts == 1
    await second.documents.close()


async def test_the_extractor_records_its_own_failure(engine):
    """Nothing has to remember to call it: a pass that could not read a
    record leaves that on the record's own receipt."""
    from scone_memory import FakeChat
    from scone_memory.ingestion.distill import DistillError, Distiller

    job = await engine.ingest_batch("alpha", records("something to read"))
    with pytest.raises(DistillError):
        # The pass tells its caller by raising; the receipt is written either way.
        await Distiller(engine, FakeChat(["not json at all"]), max_attempts=1).distill_pending("alpha")

    item = (await engine.job("alpha", job.job_id)).items[0]
    assert item.state == "failed" and item.attempts >= 1 and item.error
    assert item.searchable_at and item.consolidated_at is None
