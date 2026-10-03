"""Scheduled forgetting: a memory that carries the time it is to be forgotten.

``remember(..., forget_after=...)`` takes an RFC 3339 time, a bare date or a
duration from the engine's clock, and keeps the resolved time on the
episode's metadata under ``forget_after``. A time already past, or one that
cannot be read, is refused at the write.

Two things then hold. Recall never returns a passage whose memory is past its
time, even before anything has swept it: the read filters. And
``forget_due`` forgets what is due through the ordinary forget, so chunks,
vectors, attachments and (when asked) claims go exactly as a manual forget
takes them, and the report says what went and why. A sweep is bounded per
call and says when its bounds bit.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import Gone, InvalidInput
from scone_memory.ingestion.records import Record
from scone_memory.memory import scheduled_forget
from scone_memory.observability.events import InMemoryEventLog
from scone_memory.retrieval.reranking import RerankScore
from scone_memory.testing import Clock

NOW = "2026-09-15T12:00:00.000Z"


@pytest.fixture
async def memory():
    clock = Clock(NOW)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(), clock=clock).open()
    engine.test_clock = clock
    yield engine
    await engine.close()


def at(engine, when: str) -> None:
    engine.test_clock.now = when


# -- the write -------------------------------------------------------------


async def test_an_rfc3339_time_is_kept_on_the_episode_as_utc(memory):
    added = await memory.remember("s", "the door code is 4411", forget_after="2026-09-16T14:00:00+02:00")
    assert added.forget_after == "2026-09-16T12:00:00.000Z"
    episode = await memory.episode("s", added.episode_id)
    assert episode.metadata["forget_after"] == "2026-09-16T12:00:00.000Z"


async def test_a_duration_counts_from_the_engine_clock(memory):
    added = await memory.remember("s", "the parking spot is B12", forget_after="1d12h")
    assert added.forget_after == "2026-09-17T00:00:00.000Z"
    assert (await memory.remember("s", "a note for ninety minutes", forget_after="90m")).forget_after == "2026-09-15T13:30:00.000Z"
    assert (await memory.remember("s", "a note for two weeks", forget_after="2w")).forget_after == "2026-09-29T12:00:00.000Z"
    assert (await memory.remember("s", "a note for a minute", forget_after="60s")).forget_after == "2026-09-15T12:01:00.000Z"


async def test_a_bare_date_means_midnight_utc(memory):
    assert (await memory.remember("s", "a note until October", forget_after="2026-10-01")).forget_after == "2026-10-01T00:00:00.000Z"


@pytest.mark.parametrize("past", ["2026-09-15T11:59:59Z", NOW, "2020-01-01"], ids=["a-second-ago", "now", "years-ago"])
async def test_a_time_already_past_is_refused_and_nothing_is_stored(memory, past):
    with pytest.raises(InvalidInput, match="forget_after .* not after"):
        await memory.remember("s", "too late to schedule", forget_after=past)
    assert (await memory.status("s")).episodes == 0


@pytest.mark.parametrize("bad", ["tomorrow", "10", "5y", "0d", "d", "1.5d", "-3d", "2026-13-01", "36526d", "1d" * 40, "", 5],
                         ids=["word", "bare-number", "unknown-unit", "zero", "no-number", "fraction", "negative",
                              "bad-month", "past-the-horizon", "too-long", "empty", "not-text"])
async def test_a_time_that_cannot_be_read_is_refused(memory, bad):
    with pytest.raises(InvalidInput, match="forget_after"):
        await memory.remember("s", "a malformed schedule", forget_after=bad)
    assert (await memory.status("s")).episodes == 0


async def test_the_longest_duration_is_accepted(memory):
    added = await memory.remember("s", "a very long schedule", forget_after=f"{scheduled_forget.MAX_DURATION_DAYS}d")
    assert added.forget_after is not None and added.forget_after.startswith("2126-")


async def test_metadata_carrying_forget_after_is_held_to_the_same_rules(memory):
    added = await memory.remember("s", "scheduled through metadata", metadata={"forget_after": "2026-09-20"})
    assert added.forget_after == "2026-09-20T00:00:00.000Z"
    with pytest.raises(InvalidInput, match="forget_after"):
        await memory.remember("s", "a stale metadata schedule", metadata={"forget_after": "2026-01-01"})
    with pytest.raises(InvalidInput, match="metadata says forget_after"):
        await memory.remember("s", "two schedules that disagree", metadata={"forget_after": "2026-09-20"},
                              forget_after="2026-09-21")
    agreeing = await memory.remember("s", "two schedules that agree", metadata={"forget_after": "2026-09-20T00:00:00Z"},
                                     forget_after="2026-09-20")
    assert agreeing.forget_after == "2026-09-20T00:00:00.000Z"


async def test_the_schedule_counts_against_the_metadata_key_limit(memory):
    from scone_memory.core.validation import MAX_METADATA_KEYS

    full = {f"k{n}": "v" for n in range(MAX_METADATA_KEYS)}
    await memory.remember("s", "a full metadata map", metadata=full)
    with pytest.raises(InvalidInput, match="forget_after"):
        await memory.remember("s", "one key too many", metadata=full, forget_after="1d")


async def test_an_unscheduled_write_says_so(memory):
    assert (await memory.remember("s", "kept until someone forgets it")).forget_after is None


async def test_a_duplicate_reports_the_schedule_the_stored_episode_holds(memory):
    first = await memory.remember("s", "the same words twice")
    again = await memory.remember("s", "the same words twice", forget_after="1d")
    assert again.outcome == "duplicate" and again.episode_id == first.episode_id
    assert again.forget_after is None, "the write changed nothing, and the receipt must not say it scheduled anything"
    scheduled = await memory.remember("s", "scheduled words", forget_after="3d")
    plain_again = await memory.remember("s", "scheduled words")
    assert plain_again.outcome == "duplicate" and plain_again.forget_after == scheduled.forget_after == "2026-09-18T12:00:00.000Z"
    [one, two] = await memory.remember_many("s", [Record("twin in a batch", forget_after="2d"),
                                                  Record("twin in a batch", forget_after="5d")])
    assert two.outcome == "duplicate" and two.forget_after == one.forget_after == "2026-09-17T12:00:00.000Z"


async def test_writing_the_same_words_over_an_overdue_memory_stores_them_afresh(memory):
    first = await memory.remember("s", "the words come back", forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    again = await memory.remember("s", "the words come back")
    assert again.outcome == "accepted" and again.episode_id != first.episode_id, \
        "an overdue memory must not swallow a new write, or the next sweep takes the new words with it"
    with pytest.raises(Gone):
        await memory.episode("s", first.episode_id)
    assert (await memory.forget_due("s")).due == 0
    assert (await memory.episode("s", again.episode_id)).content == "the words come back"


async def test_writing_over_an_overdue_memory_keeps_the_attachments_the_new_write_links(memory):
    # The caller stores the bytes before the write; the overdue memory's forget
    # must not release them from under the episode that is about to carry them.
    original = await memory.attach("s", b"the scanned receipt", "text/plain", filename="receipt.txt")
    first = await memory.remember("s", "a receipt kept for an hour", attachment_ids=[original.attachment_id],
                                  forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    again_bytes = await memory.attach("s", b"the scanned receipt", "text/plain", filename="receipt.txt")
    again = await memory.remember("s", "a receipt kept for an hour", attachment_ids=[again_bytes.attachment_id])
    assert again.forgot_overdue is not None and again.forgot_overdue.episode_id == first.episode_id
    stored = await memory.episode("s", again.episode_id)
    assert [a.attachment_id for a in stored.attachments] == [original.attachment_id]
    assert (await memory.attachment("s", original.attachment_id))[1] == b"the scanned receipt"


async def test_a_write_over_a_memory_not_yet_due_reads_no_attachment_bytes(memory, monkeypatch):
    original = await memory.attach("s", b"a large scan", "text/plain", filename="scan.txt")
    reads: list[str] = []
    real_get = memory.blobs.get

    async def counted(space, attachment_id):
        reads.append(attachment_id)
        return await real_get(space, attachment_id)

    monkeypatch.setattr(memory.blobs, "get", counted)
    await memory.remember("s", "a scan kept for a day", attachment_ids=[original.attachment_id], forget_after="1d")
    again = await memory.remember("s", "a scan kept for a day", attachment_ids=[original.attachment_id])
    assert again.outcome == "duplicate"
    assert reads == [original.attachment_id] * 2, "only linking reads the hold; no bytes are read to be put back"
    lookups: list[str] = []
    real_lookup = memory.documents.episode_by_hash

    async def looked_up(space, digest):
        lookups.append(digest)
        return await real_lookup(space, digest)

    monkeypatch.setattr(memory.documents, "episode_by_hash", looked_up)
    await memory.remember("s", "a write with nothing to link")
    assert len(lookups) == 1, "a write that links nothing does not look its identity up ahead of the write"


async def test_a_refused_batch_forgets_no_overdue_memory_early(memory):
    first = await memory.remember("s", "words the batch would replace", forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    with pytest.raises(InvalidInput, match="forget_after"):
        await memory.remember_many("s", [Record("words the batch would replace"), Record("refused", forget_after="2020-01-01")])
    assert await memory.documents.get_episode("s", first.episode_id) is not None, "nothing of a refused batch happens"
    assert await memory.tombstone("s", first.episode_id) is None


async def test_replacing_a_keyed_record_over_an_overdue_one_stores_it_afresh(memory):
    first = await memory.remember("s", "keyed words", dedup_key="k", forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    again = await memory.remember("s", "keyed words", dedup_key="k", replace=True)
    assert again.outcome == "updated" and again.episode_id != first.episode_id
    assert (await memory.episode_by_key("s", "k")).episode_id == again.episode_id


async def test_replacing_with_the_same_words_reports_the_schedule_the_episode_holds(memory):
    first = await memory.remember("s", "keyed schedule words", dedup_key="k", forget_after="3d")
    again = await memory.remember("s", "keyed schedule words", dedup_key="k", replace=True)
    assert again.outcome == "duplicate" and again.episode_id == first.episode_id
    assert again.forget_after == first.forget_after == "2026-09-18T12:00:00.000Z", \
        "a duplicate's receipt says the schedule the stored episode holds, as a plain write's does"


async def test_a_write_over_an_overdue_memory_names_the_memory_it_forgot(memory):
    first = await memory.remember("s", "secret words", forget_after="1h")
    plain = await memory.remember("s", "plain words")
    at(memory, "2026-09-15T14:00:00.000Z")
    again = await memory.remember("s", "secret words")
    assert again.forgot_overdue is not None and again.forgot_overdue.episode_id == first.episode_id
    assert again.forgot_overdue.forget_after == "2026-09-15T13:00:00.000Z" and again.forgot_overdue.outcome == "forgotten"
    assert again.forgot_overdue.receipt is not None and again.forgot_overdue.receipt.episode_id == first.episode_id
    assert "had passed" in again.forgot_overdue.reason
    [batch_twin, fresh] = await memory.remember_many("s", [Record("plain words"), Record("new words")])
    assert batch_twin.forgot_overdue is None and fresh.forgot_overdue is None, "only a write that forgot something says so"
    assert batch_twin.episode_id == plain.episode_id
    later = await memory.remember("s", "words due later", forget_after="1h")
    at(memory, "2026-09-15T16:00:00.000Z")
    [first_in_batch, over_later] = await memory.remember_many("s", [Record("a fresh first record"), Record("words due later")])
    assert first_in_batch.forgot_overdue is None, "the receipt that names the forgotten memory is the write that replaced it"
    assert over_later.forgot_overdue is not None and over_later.forgot_overdue.episode_id == later.episode_id


async def test_a_partial_batch_answers_a_bad_schedule_where_it_was_asked(memory):
    added = await memory.remember_many("s", [Record("fine", forget_after="1d"), Record("late", forget_after="2020-01-01")],
                                       partial=True)
    assert added[0].outcome == "accepted" and added[1].outcome == "failed" and "forget_after" in added[1].reason


# -- the read, before any sweep --------------------------------------------


async def test_recall_never_returns_a_passage_past_its_time_before_the_sweep(memory):
    gone = await memory.remember("s", "the wifi password is hunter2", forget_after="1h")
    kept = await memory.remember("s", "the wifi network is called attic")
    before = await memory.recall("s", "wifi password", limit=5)
    assert {item.episode_id for item in before.items} == {gone.episode_id, kept.episode_id}
    assert before.past_forget_after is None, "nothing withheld, nothing said"
    at(memory, "2026-09-15T13:00:00.000Z")
    after = await memory.recall("s", "wifi password", limit=5)
    assert [item.episode_id for item in after.items] == [kept.episode_id]
    assert after.past_forget_after == {"withheld": 1, "episode_ids": [gone.episode_id], "at": "2026-09-15T13:00:00.000Z"}
    assert (await memory.episode("s", kept.episode_id)).content, "the other memory is untouched"
    assert (await memory.status("s")).episodes == 2, "the read withholds; only the sweep forgets"
    [event] = [e for e in await memory.events.query(space="s", kind="recall", limit=1)]
    assert event.payload["past_forget_after"] == after.past_forget_after, "the recall's evidence says what it withheld"


async def test_every_withheld_passage_is_counted(memory):
    due = await memory.remember("s", " ".join(f"Clause {n} of the tenancy agreement." for n in range(60)), forget_after="1h")
    kept = await memory.remember("s", "the tenancy agreement renews in May")
    assert len(await memory.documents.chunks_of("s", due.episode_id)) >= 3
    at(memory, "2026-09-15T13:00:00.000Z")
    result = await memory.recall("s", "tenancy agreement clause", limit=5)
    assert [i.episode_id for i in result.items] == [kept.episode_id]
    assert result.past_forget_after["withheld"] == 2, "the two passages the episode cap let through, each withheld"
    assert result.past_forget_after["episode_ids"] == [due.episode_id]


async def test_the_withheld_passage_gives_its_place_to_the_next_one(memory):
    due = [await memory.remember("s", f"ticket {n} for the concert", forget_after="1h") for n in range(3)]
    kept = [await memory.remember("s", f"ticket {n} for the play") for n in range(3)]
    at(memory, "2026-09-15T13:00:00.000Z")
    result = await memory.recall("s", "ticket for the concert", limit=3)
    assert sorted(item.episode_id for item in result.items) == sorted(a.episode_id for a in kept), \
        "the limit is filled from what may be returned, not cut first and emptied after"
    assert result.past_forget_after["withheld"] == 3
    assert sorted(result.past_forget_after["episode_ids"]) == sorted(a.episode_id for a in due)


async def test_the_boundary_is_the_scheduled_instant(memory):
    added = await memory.remember("s", "a meeting room booking", forget_after="2026-09-15T13:00:00Z")
    at(memory, "2026-09-15T12:59:59.999Z")
    assert [i.episode_id for i in (await memory.recall("s", "meeting room booking")).items] == [added.episode_id]
    at(memory, "2026-09-15T13:00:00.000Z")
    assert (await memory.recall("s", "meeting room booking")).items == []


async def test_an_overdue_memory_reads_as_gone_by_id_and_by_key(memory):
    added = await memory.remember("s", "a secret with a key", dedup_key="secret", forget_after="1h")
    at(memory, "2026-09-15T13:00:00.000Z")
    with pytest.raises(Gone, match="due to be forgotten"):
        await memory.episode("s", added.episode_id)
    with pytest.raises(Gone, match="due to be forgotten"):
        await memory.episode_by_key("s", "secret")


async def test_filling_the_answer_reads_no_more_episodes_than_recall_did(memory):
    for n in range(8):
        await memory.remember("s", f"orchard apple harvest note {n}")
    reads = []
    original = memory.documents.get_episode

    async def counted(space, episode_id):
        reads.append(episode_id)
        return await original(space, episode_id)

    memory.documents.get_episode = counted
    result = await memory.recall("s", "orchard apple harvest", limit=2)
    assert len(result.items) == 2 and len(reads) == 2, "the answer's own episodes, and no candidate past it"


async def test_a_reranker_chooses_among_every_candidate_that_is_not_due(memory):
    from unittest.mock import AsyncMock

    class Answer:
        async def rerank(self, query, candidates):
            return [RerankScore(c.chunk_id, 10.0 if "ANSWER" in c.text else 0.0) for c in candidates]

    due = await memory.remember("s", "NOISE the ferry timetable, due", forget_after="1h")
    plain = await memory.remember("s", "NOISE the ferry timetable, kept")
    answer = await memory.remember("s", "ANSWER the ferry timetable, kept")
    order = [(await memory.documents.chunks_of("s", e.episode_id))[0].chunk_id for e in (due, plain, answer)]
    memory.vectors.search = AsyncMock(side_effect=lambda *args, **kwargs: [(cid, 0.9 - i / 100) for i, cid in enumerate(order)])
    memory.reranker = Answer()
    at(memory, "2026-09-15T13:00:00.000Z")
    result = await memory.recall("s", "ferry timetable", limit=1, lanes=("vector",))
    assert [i.episode_id for i in result.items] == [answer.episode_id]


async def test_a_reranked_recall_withholds_too(memory):
    class Reverse:
        async def rerank(self, query, candidates):
            return [RerankScore(c.chunk_id, float(n)) for n, c in enumerate(candidates)]

    gone = await memory.remember("s", "the locker code is 9021", forget_after="1h")
    kept = await memory.remember("s", "the locker is on floor two")
    memory.reranker = Reverse()
    at(memory, "2026-09-15T13:00:00.000Z")
    result = await memory.recall("s", "locker code", limit=5)
    assert [i.episode_id for i in result.items] == [kept.episode_id]
    assert result.past_forget_after["episode_ids"] == [gone.episode_id]


async def test_a_summary_of_an_overdue_document_brings_nothing_of_it():
    from scone_memory.retrieval.summary_expand import expand_summaries
    from tests.retrieval.test_summary_expand import PARTS, forged_summary

    clock = Clock(NOW)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=100,
                                clock=clock).open()
    try:
        added = await engine.remember("s", "\n\n".join(PARTS), kind="file", source="town.md", forget_after="1h")
        chunks = await engine.documents.chunks_of("s", added.episode_id)
        summary = await forged_summary(engine, added.episode_id, chunks)
        assert (await expand_summaries(engine, "s", summary)).chunks_added == 1
        clock.now = "2026-09-15T13:00:00.000Z"
        due = await expand_summaries(engine, "s", summary, mode="replace")
        assert due.chunks_added == 0 and due.items == tuple(summary), "a document past its time is served by no door"
        assert [r["reason"] for r in due.refused] == ["source_gone"]
    finally:
        await engine.close()


async def test_a_listing_by_metadata_leaves_an_overdue_turn_out(memory):
    gone = await memory.remember("s", "turn one", metadata={"session_id": "a"}, forget_after="1h")
    kept = await memory.remember("s", "turn two", metadata={"session_id": "a"})
    at(memory, "2026-09-15T13:00:00.000Z")
    assert [e.episode_id for e in await memory.episodes("s", {"session_id": "a"})] == [kept.episode_id]
    assert gone.episode_id not in [e.episode_id for e in await memory.episodes("s", {"session_id": "a"}, limit=5)]


async def test_the_overview_leaves_an_overdue_memory_out_and_counts_it(memory):
    kept = await memory.remember("s", "the router lives in the hall")
    await memory.remember("s", "the wifi password is hunter2", forget_after="1h")
    at(memory, "2026-09-15T13:00:00.000Z")
    shown = await memory.overview("s")
    assert [item.episode_id for item in shown.items] == [kept.episode_id], "an overdue memory must not reach a model by overview"
    assert shown.past_forget_after == 1
    filtered = await memory.overview("s", kind="note", limit=1)
    assert [item.episode_id for item in filtered.items] == [kept.episode_id] and filtered.past_forget_after == 1
    assert (await memory.overview("s", kind="file")).past_forget_after == 0, "only a memory the filters would show is counted"


async def test_the_source_listing_leaves_an_overdue_memory_out_and_still_fills_its_page(memory):
    work = await memory.remember("s", "a note about work", metadata={"topic": "work"})
    older = await memory.remember("s", "an older kept note", metadata={"topic": "home"})
    kept = await memory.remember("s", "the router lives in the hall", metadata={"topic": "home"})
    await memory.remember("s", "the wifi password is hunter2", metadata={"topic": "home"}, forget_after="1h")
    at(memory, "2026-09-15T13:00:00.000Z")
    page = await memory.source_page("s", limit=1)
    assert [e.episode_id for e in page.episodes] == [kept.episode_id] and page.past_forget_after == 1
    assert page.has_more and page.next_before == kept.episode_id
    rest = await memory.source_page("s", limit=1, before=page.next_before)
    assert [e.episode_id for e in rest.episodes] == [older.episode_id] and rest.past_forget_after == 0
    narrowed = await memory.source_page("s", conditions={"field": "topic", "is": "home"})
    assert [e.episode_id for e in narrowed.episodes] == [kept.episode_id, older.episode_id]
    assert narrowed.past_forget_after == 1
    plain = await memory.source_page("s", limit=5)
    assert [e.episode_id for e in plain.episodes] == [kept.episode_id, older.episode_id, work.episode_id]
    assert not plain.has_more


async def test_forgetting_by_filter_still_selects_an_overdue_memory(memory):
    due = await memory.remember("s", "a synced note past its time", source="sync/a.md", forget_after="1h")
    at(memory, "2026-09-15T13:00:00.000Z")
    preview = await memory.forget_matching("s", source_prefix="sync/")
    assert preview.episode_ids == [due.episode_id], "forget reaches an overdue memory, and so does forgetting by filter"


# -- the sweep ---------------------------------------------------------------


async def test_the_sweep_forgets_what_is_due_and_leaves_what_is_not(memory):
    due = await memory.remember("s", "due in an hour", forget_after="1h")
    later = await memory.remember("s", "due in a week", forget_after="1w")
    plain = await memory.remember("s", "never scheduled")
    at(memory, "2026-09-15T13:00:00.000Z")
    report = await memory.forget_due("s")
    assert report.forgotten == [due.episode_id] and report.due == 1 and report.remaining == 0
    [item] = report.items
    assert item.episode_id == due.episode_id and item.forget_after == "2026-09-15T13:00:00.000Z"
    assert item.outcome == "forgotten" and item.receipt is not None and item.receipt.forgotten_at is not None
    assert "forget_after" in item.reason and "2026-09-15T13:00:00.000Z" in item.reason
    assert report.now == "2026-09-15T13:00:00.000Z" and report.limited is False and report.scan_complete is True
    for episode in (later, plain):
        assert (await memory.episode("s", episode.episode_id)).episode_id == episode.episode_id
    stone = await memory.tombstone("s", due.episode_id)
    assert stone is not None and stone.forgotten_at == "2026-09-15T13:00:00.000Z"
    assert (await memory.forget_due("s")).forgotten == [], "a second sweep finds nothing left to do"


async def test_the_sweep_goes_through_forget_so_derived_data_goes_with_it(memory):
    attachment = await memory.attach("s", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png", "code.png")
    due = await memory.remember("s", "Alice holds the vault key. " * 40, forget_after="1h",
                                attachment_ids=[attachment.attachment_id])
    fact = await memory.assert_fact("s", "Alice", "holds", "vault key", source_episode_id=due.episode_id,
                                    quote="Alice holds the vault key.")
    chunk_ids = [c.chunk_id for c in await memory.documents.chunks_of("s", due.episode_id)]
    assert len(chunk_ids) > 1 and set(chunk_ids) <= set(await memory.vectors.ids("s"))
    at(memory, "2026-09-15T13:00:00.000Z")
    report = await memory.forget_due("s", with_claims="exclude")
    [item] = report.items
    assert item.receipt.chunks == len(chunk_ids) and item.receipt.attachments_released == [attachment.attachment_id]
    assert item.receipt.claims_excluded == [fact.fact_id] and report.with_claims == "exclude"
    assert await memory.documents.chunks_of("s", due.episode_id) == []
    assert not set(chunk_ids) & set(await memory.vectors.ids("s")), "the vectors went with the chunks"
    assert (await memory.documents.get_fact("s", fact.fact_id)).excluded
    doctor = await memory.doctor("s")
    assert doctor.tombstones == 1 and doctor.not_inspected == []
    assert doctor.chunks_without_episode == [] and doctor.vectors_without_chunk == [] and doctor.attachments_unlinked == []
    assert doctor.facts_citing_forgotten == [fact.fact_id], "the claim stands citing a forgotten source, as after a manual forget"
    kinds = [event.kind for event in await memory.events.query(space="s", limit=50)]
    assert "forget" in kinds and "forget_due" in kinds


async def test_the_sweep_keeps_claims_by_default_as_forget_does(memory):
    due = await memory.remember("s", "Bob owns the red bike", forget_after="1h")
    fact = await memory.assert_fact("s", "Bob", "owns", "red bike", source_episode_id=due.episode_id, quote="Bob owns the red bike")
    at(memory, "2026-09-15T13:00:00.000Z")
    [item] = (await memory.forget_due("s")).items
    assert item.receipt.facts_citing == [fact.fact_id] and item.receipt.claims_excluded == []
    assert not (await memory.documents.get_fact("s", fact.fact_id)).excluded


async def test_a_pass_is_bounded_by_its_limit_and_says_so(memory):
    due = [await memory.remember("s", f"due note {n}", forget_after=f"{n + 1}h") for n in range(3)]
    at(memory, "2026-09-16T00:00:00.000Z")
    first = await memory.forget_due("s", limit=1)
    assert first.forgotten == [due[0].episode_id], "the most overdue goes first"
    assert first.due == 3 and first.limited is True and first.remaining == 2
    second = await memory.forget_due("s", limit=2)
    assert second.forgotten == [due[1].episode_id, due[2].episode_id] and second.limited is False and second.remaining == 0


async def test_a_pass_is_bounded_by_its_walk_and_says_where_to_resume(memory, monkeypatch):
    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 4)
    monkeypatch.setattr(scheduled_forget, "PAGE", 2)
    oldest = await memory.remember("s", "the oldest note, due", forget_after="1h")
    for n in range(5):
        await memory.remember("s", f"a newer note {n}")
    at(memory, "2026-09-15T13:00:00.000Z")
    cut = await memory.forget_due("s")
    assert cut.scan_complete is False and cut.scanned == 4 and cut.forgotten == [] and cut.resume_before is not None
    resumed = await memory.forget_due("s", before=cut.resume_before)
    assert resumed.forgotten == [oldest.episode_id] and resumed.scan_complete is True and resumed.resume_before is None
    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 3)
    odd = await memory.forget_due("s")
    assert odd.scanned == 3 and odd.scan_complete is False
    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 5)
    exact = await memory.forget_due("s")
    assert exact.scanned == 5 and exact.scan_complete is True, "a walk that ends on its bound at the end of the space is complete"
    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 100)
    pages = []
    walk = memory.documents.page_episodes

    async def counted(*args):
        pages.append(args)
        return await walk(*args)

    memory.documents.page_episodes = counted
    short = await memory.forget_due("s")
    assert short.scan_complete is True and len(pages) == 3, "a short page is the end: no read past it"


async def test_a_dry_run_names_what_would_go_and_forgets_nothing(memory):
    due = await memory.remember("s", "a dry run candidate", forget_after="1h")
    at(memory, "2026-09-15T13:00:00.000Z")
    preview = await memory.forget_due("s", dry_run=True)
    assert preview.forgotten == [] and [i.episode_id for i in preview.items] == [due.episode_id]
    assert preview.items[0].outcome == "would_forget" and preview.items[0].receipt is None and preview.remaining == 1
    assert (await memory.status("s")).episodes == 1
    assert [e for e in await memory.events.query(space="s", limit=50) if e.kind == "forget_due"] == [], "a dry run records nothing"


async def test_the_sweep_cannot_be_told_it_is_later_than_it_is(memory):
    await memory.remember("s", "not due yet", forget_after="1d")
    with pytest.raises(InvalidInput, match="later than the engine clock"):
        await memory.forget_due("s", now="2026-09-17T00:00:00Z")
    assert (await memory.status("s")).episodes == 1
    earlier = await memory.forget_due("s", now="2026-09-15T00:00:00Z")
    assert earlier.forgotten == [] and earlier.now == "2026-09-15T00:00:00.000Z"


@pytest.mark.parametrize("limit", [0, 1001, True, "5"], ids=["zero", "over", "bool", "text"])
async def test_a_limit_outside_its_range_is_refused(memory, limit):
    with pytest.raises(InvalidInput, match="limit"):
        await memory.forget_due("s", limit=limit)


async def test_other_arguments_a_sweep_cannot_mean_are_refused(memory):
    with pytest.raises(InvalidInput, match="with_claims"):
        await memory.forget_due("s", with_claims="all", dry_run=True)
    with pytest.raises(InvalidInput, match="before"):
        await memory.forget_due("s", before=0)
    with pytest.raises(InvalidInput, match="now"):
        await memory.forget_due("s", now=1234)
    memory.documents.page_episodes = None
    with pytest.raises(InvalidInput, match="source inventory"):
        await memory.forget_due("s")


async def test_an_episode_gone_before_its_forget_ran_is_skipped_and_said(memory):
    first = await memory.remember("s", "raced by a manual forget", forget_after="1h")
    second = await memory.remember("s", "swept normally", forget_after="2h")
    original = memory.forget

    async def racing(space, episode_id, **kwargs):
        if episode_id == first.episode_id:
            await original(space, episode_id)
        return await original(space, episode_id, **kwargs)

    memory.forget = racing
    at(memory, "2026-09-15T15:00:00.000Z")
    report = await memory.forget_due("s")
    assert report.forgotten == [second.episode_id] and report.remaining == 0
    assert report.items[0].outcome == "skipped" and "already forgotten" in report.items[0].reason


async def test_a_schedule_the_sweep_cannot_read_is_named_not_forgotten(memory):
    added = await memory.remember("s", "written before the schedule existed")
    stored = await memory.documents.get_episode("s", added.episode_id)
    memory.documents._episodes[added.episode_id] = stored.model_copy(update={"metadata": {"forget_after": "someday"}})
    at(memory, "2027-01-01T00:00:00.000Z")
    report = await memory.forget_due("s")
    assert report.forgotten == [] and report.unreadable == [added.episode_id]
    assert [i.episode_id for i in (await memory.recall("s", "written before the schedule")).items] == [added.episode_id]


# -- moving a space ------------------------------------------------------------


async def test_export_and_import_keep_the_schedule(memory):
    added = await memory.remember("s", "a schedule that travels", forget_after="2d")
    dump = [record async for record in memory.export("s")]
    summary = await memory.import_records("t", dump)
    assert summary.episodes == 1
    [moved] = (await memory.source_page("t")).episodes
    assert moved.metadata["forget_after"] == added.forget_after
    at(memory, "2026-09-18T00:00:00.000Z")
    assert (await memory.forget_due("t")).forgotten == [moved.episode_id]


async def test_an_archive_holding_an_overdue_memory_imports_the_rest_and_counts_it(memory):
    due = await memory.remember("s", "overdue in the archive", forget_after="1h")
    await memory.remember("s", "kept in the archive")
    dump = [record async for record in memory.export("s")]
    at(memory, "2026-09-15T13:00:00.000Z")
    summary = await memory.import_records("t", dump)
    assert summary.episodes == 1 and summary.past_forget_after == 1
    assert summary.record()["past_forget_after"] == 1
    assert [e.content for e in (await memory.source_page("t")).episodes] == ["kept in the archive"]
    assert "past_forget_after" not in (await memory.import_records("u", dump[:1])).record(), "a legacy receipt stays as it was"
    assert due.episode_id


async def test_merging_a_space_leaves_an_overdue_memory_behind(memory):
    await memory.remember("s", "overdue at the merge", forget_after="1h")
    kept = await memory.remember("s", "moves with the merge", forget_after="1w")
    at(memory, "2026-09-15T13:00:00.000Z")
    preview = await memory.merge_space("s", into="t", preview=True)
    assert preview.episodes == 1 and preview.past_forget_after == 1 and preview.tombstoned == 0
    done = await memory.merge_space("s", into="t", confirm="s")
    assert done.moved is True
    [moved] = (await memory.source_page("t")).episodes
    assert moved.content == "moves with the merge" and moved.metadata["forget_after"] == kept.forget_after


async def test_a_memory_that_comes_due_during_a_merge_refuses_it_and_the_source_stays(memory):
    await memory.remember("s", "due in an hour, mid merge", forget_after="1h")
    original = memory.import_records

    async def late(space, records, **kwargs):
        at(memory, "2026-09-15T13:30:00.000Z")
        return await original(space, records, **kwargs)

    memory.import_records = late
    with pytest.raises(InvalidInput, match="came due"):
        await memory.merge_space("s", into="t", confirm="s")
    assert (await memory.status("s")).episodes == 1


async def test_a_schedule_that_comes_due_after_the_copy_does_not_fail_the_merge_check(memory):
    await memory.remember("s", "due in an hour, just after the copy", forget_after="1h")
    original = memory.import_records

    async def then_later(space, records, **kwargs):
        summary = await original(space, records, **kwargs)
        at(memory, "2026-09-15T13:30:00.000Z")
        return summary

    memory.import_records = then_later
    done = await memory.merge_space("s", into="t", confirm="s")
    assert done.moved is True and done.episodes == 1


# -- the worker ----------------------------------------------------------------


async def test_the_worker_sweeps_what_is_due_on_its_pass(memory):
    from scone_memory.ingestion.worker import ConsolidationWorker

    due = await memory.remember("s", "swept by the worker", forget_after="1h")
    worker = ConsolidationWorker(memory, None, ["s"], retention={"conversation": 30})
    at(memory, "2026-09-15T13:00:00.000Z")
    report = await worker.run_once("s")
    assert report.forgotten_due == 1 and report.error is None and report.forget_due_limited is False
    with pytest.raises(Gone):
        await memory.episode("s", due.episode_id)
    for n in range(2):
        await memory.remember("s", f"worker batch note {n}", forget_after="1h")
    at(memory, "2026-09-15T15:00:00.000Z")
    worker.batch = 1
    small = await worker.run_once("s")
    assert small.forgotten_due == 1 and small.forget_due_limited is True, "the pass says its limit bit"
    worker.batch = 5000
    large = await worker.run_once("s")
    assert large.error is None and large.forgotten_due == 1, "a batch past one pass's limit is held to it, not refused"


async def test_a_sweep_that_fails_is_the_pass_error(memory):
    from scone_memory.ingestion.worker import ConsolidationWorker

    async def broken(*args, **kwargs):
        raise InvalidInput("the store cannot walk its sources")

    memory.forget_due = broken
    report = await ConsolidationWorker(memory, None, ["s"], retention={"conversation": 30}).run_once("s")
    assert report.error == "InvalidInput: the store cannot walk its sources"


async def test_the_worker_walks_on_to_older_episodes_across_passes(memory, monkeypatch, caplog):
    import logging

    from scone_memory.ingestion.worker import ConsolidationWorker

    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 4)
    monkeypatch.setattr(scheduled_forget, "PAGE", 2)
    caplog.set_level(logging.INFO, logger="scone_memory.ingestion.worker")
    oldest = await memory.remember("s", "the oldest note, due", forget_after="1h")
    for n in range(6):
        await memory.remember("s", f"a newer unscheduled note {n}")
    at(memory, "2026-09-15T13:00:00.000Z")
    worker = ConsolidationWorker(memory, None, ["s"], retention={"conversation": 30})
    first = await worker.run_once("s")
    assert first.forgotten_due == 0 and first.forget_due_scan_cut is True and first.forget_due_limited is False
    [finished] = [r for r in caplog.records if getattr(r, "event", None) == "consolidation.finished"]
    assert finished.forget_due_scan_cut is True and finished.forget_due_limited is False
    second = await worker.run_once("s")
    assert second.forgotten_due == 1 and second.forget_due_scan_cut is False, "the next pass walks on from where the last stopped"
    assert await memory.documents.get_episode("s", oldest.episode_id) is None
    newest = await memory.remember("s", "a new note, due soon", forget_after="1s")
    at(memory, "2026-09-15T13:00:01.000Z")
    third = await worker.run_once("s")
    assert third.forgotten_due == 1 and await memory.documents.get_episode("s", newest.episode_id) is None, \
        "a walk that reached the oldest starts again from the newest"


async def test_the_worker_drains_a_window_before_walking_on(memory, monkeypatch):
    from scone_memory.ingestion.worker import ConsolidationWorker

    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 4)
    monkeypatch.setattr(scheduled_forget, "PAGE", 2)
    for n in range(3):
        await memory.remember("s", f"an old unscheduled note {n}")
    due = [await memory.remember("s", f"a due note {n}", forget_after="1h") for n in range(3)]
    at(memory, "2026-09-15T13:00:00.000Z")
    worker = ConsolidationWorker(memory, None, ["s"], retention={"conversation": 30}, batch=1)
    reports = [await worker.run_once("s") for _ in range(3)]
    assert [r.forgotten_due for r in reports] == [1, 1, 1] and reports[0].forget_due_limited is True
    assert [await memory.documents.get_episode("s", d.episode_id) for d in due] == [None, None, None], \
        "a limited pass leaves the rest of its window for the next pass, not for the next lap"


async def test_a_store_without_source_inventory_still_expires_and_says_it_did_not_sweep(memory):
    from scone_memory.ingestion.worker import ConsolidationWorker

    await memory.remember("s", "an old note", kind="note", created_at="2026-01-01T00:00:00.000Z")

    class NoInventory(type(memory.documents)):
        page_episodes = None

    memory.documents.__class__ = NoInventory
    report = await ConsolidationWorker(memory, None, ["s"], retention={"note": 30}).run_once("s")
    assert report.error is None and report.expired == 1, "a sweep the store cannot run must not stop retention"
    assert report.forget_due_skipped is not None and "source inventory" in report.forget_due_skipped


async def test_a_sweep_that_hits_a_transport_error_is_recorded_and_the_pass_goes_on(memory):
    from scone_memory.ingestion.worker import ConsolidationWorker

    await memory.remember("s", "an old note", kind="note", created_at="2026-01-01T00:00:00.000Z")

    async def hiccup(*args, **kwargs):
        raise ConnectionError("store hiccup with private detail")

    memory.documents.page_episodes = hiccup
    report = await ConsolidationWorker(memory, None, ["s"], retention={"note": 30}).run_once("s")
    assert report.error == "ConnectionError", "a worker must not die on a transport hiccup, and no error text is kept"
    assert report.expired == 1, "a failed sweep does not stop retention"


async def test_a_sweep_that_takes_nothing_records_no_event(memory):
    await memory.remember("s", "not due for a day", forget_after="1d")
    empty = await memory.forget_due("s")
    assert empty.items == []
    assert [e for e in await memory.events.query(space="s", limit=50) if e.kind == "forget_due"] == [], \
        "a timer that sweeps every pass must not fill the log with empty passes"
    at(memory, "2026-09-16T12:00:00.000Z")
    await memory.forget_due("s")
    assert len([e for e in await memory.events.query(space="s", limit=50) if e.kind == "forget_due"]) == 1


def test_the_sync_engine_sweeps_too():
    from scone_memory import SyncMemoryEngine

    clock = Clock(NOW)
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock)) as sync:
        added = sync.remember("s", "a synchronous schedule", forget_after="1h")
        clock.now = "2026-09-15T13:00:00.000Z"
        preview = sync.forget_due("s", dry_run=True)
        assert preview.forgotten == [] and [i.episode_id for i in preview.items] == [added.episode_id]
        assert sync.forget_due("s").forgotten == [added.episode_id]


# -- every store ---------------------------------------------------------------


async def test_every_store_keeps_the_schedule_withholds_it_and_sweeps_it(engine):
    """No store learns anything new: the schedule is metadata, the sweep walks
    the source inventory every store has, and the forget is the ordinary one."""
    engine.test_clock.now = "2026-09-15T12:00:00.000Z"
    due = await engine.remember("default", "the storage unit combination is 5519", forget_after="1h")
    kept = await engine.remember("default", "the storage unit is at the north depot")
    stored = await engine.documents.get_episode("default", due.episode_id)
    assert stored.metadata["forget_after"] == "2026-09-15T13:00:00.000Z"
    engine.test_clock.now = "2026-09-15T13:00:00.000Z"
    found = await engine.recall("default", "storage unit combination", limit=5)
    assert due.episode_id not in [i.episode_id for i in found.items]
    assert kept.episode_id in [i.episode_id for i in found.items]
    report = await engine.forget_due("default")
    assert report.forgotten == [due.episode_id] and report.scan_complete is True
    assert await engine.documents.chunks_of("default", due.episode_id) == []
    assert await engine.tombstone("default", due.episode_id) is not None
    assert [e.episode_id for e in (await engine.source_page("default")).episodes] == [kept.episode_id]


async def test_consolidation_does_not_read_an_overdue_memory(memory):
    from scone_memory.ingestion.distill import Distiller
    from scone_memory.providers.llm import FakeChat

    await memory.remember("s", "Carol keeps the spare house key", forget_after="1h")
    await memory.remember("s", "Dave waters the plants on Sunday")
    chat = FakeChat(['{"facts": []}'] * 3)
    at(memory, "2026-09-15T13:00:00.000Z")
    await Distiller(memory, chat).distill_pending("s")
    read = " ".join(user for _, user in chat.calls)
    assert "Dave waters" in read and "Carol" not in read, "a model must not be handed what is due to be forgotten"


async def test_the_worker_sweeps_even_when_consolidation_fails(memory):
    from scone_memory.ingestion.distill import Distiller
    from scone_memory.ingestion.worker import ConsolidationWorker
    from scone_memory.providers.llm import FakeChat

    due = await memory.remember("s", "swept although the model is down", forget_after="1h")
    await memory.remember("s", "a pending note the model cannot read")
    worker = ConsolidationWorker(memory, Distiller(memory, FakeChat([])), ["s"])
    at(memory, "2026-09-15T13:00:00.000Z")
    report = await worker.run_once("s")
    assert report.error is not None and report.forgotten_due == 1
    with pytest.raises(Gone):
        await memory.episode("s", due.episode_id)
