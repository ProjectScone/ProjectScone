"""The behavioural contract every backend pair must satisfy.

These are the tests the in-process, SQLite, MongoDB and Qdrant adapters
pass. Import them with ``from scone_memory.testing.contract import *``
and supply an ``engine`` fixture; see ``scone_memory.testing``.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ..errors import InvalidInput, NotFound

async def test_remember_then_recall_finds_it(engine):
    added = await engine.remember("default", "Moved to Lisbon in March; the flat is on Rua Augusta")
    await engine.remember("default", "Quarterly revenue exceeded analyst expectations")
    result = await engine.recall("default", "which street is my Lisbon flat on", limit=1)
    assert [i.episode_id for i in result.items] == [added.episode_id]
    assert result.items[0].score == 1.0
    assert result.degraded == []


async def test_same_content_is_deduplicated(engine):
    first = await engine.remember("default", "I take my coffee black")
    again = await engine.remember("default", "I take my coffee black")
    assert again.deduplicated and again.episode_id == first.episode_id
    assert (await engine.status("default")).episodes == 1


async def test_spaces_do_not_leak(engine):
    await engine.remember("alpha", "the password hint is bluebird")
    result = await engine.recall("beta", "password hint bluebird")
    assert result.items == []


async def test_as_of_hides_what_happened_later(engine):
    await engine.remember("default", "booked a dentist appointment", created_at="2024-01-10")
    late = await engine.remember("default", "dentist appointment went fine", created_at="2024-03-10")
    result = await engine.recall("default", "dentist appointment", as_of="2024-02-01")
    assert late.episode_id not in {i.episode_id for i in result.items}
    assert len(result.items) == 1


async def test_tag_filter_requires_every_tag(engine):
    both = await engine.remember("default", "sprint planning for the api", tags=["work", "planning"])
    await engine.remember("default", "sprint planning for the garden", tags=["home", "planning"])
    result = await engine.recall("default", "sprint planning", tags=["work", "planning"])
    assert [i.episode_id for i in result.items] == [both.episode_id]


async def test_a_broken_vector_lane_is_reported_not_fatal(engine):
    await engine.remember("default", "the meeting moved to thursday")

    async def explode(*_args, **_kwargs):
        raise ConnectionError("qdrant unreachable")

    engine.vectors.search = explode
    result = await engine.recall("default", "when is the meeting")
    assert [i.text for i in result.items] == ["the meeting moved to thursday"]
    assert result.degraded and result.degraded[0].startswith("vectors: ConnectionError")


async def test_at_most_two_chunks_per_episode(engine):
    long = "\n\n".join(f"Paragraph {i} about the kubernetes migration and its cluster upgrade." for i in range(12))
    await engine.remember("default", long)
    other = await engine.remember("default", "one short note on the kubernetes migration")
    result = await engine.recall("default", "kubernetes migration cluster upgrade", limit=5)
    per_episode = Counter(i.episode_id for i in result.items)
    assert max(per_episode.values()) == 2
    assert other.episode_id in per_episode


async def test_forget_removes_from_recall(engine):
    added = await engine.remember("default", "temporary scratch thought about penguins")
    await engine.forget("default", added.episode_id)
    assert (await engine.recall("default", "penguins")).items == []
    with pytest.raises(NotFound):
        await engine.forget("default", added.episode_id)


async def test_bad_input_is_refused_loudly(engine):
    with pytest.raises(InvalidInput):
        await engine.remember("Bad Space!", "x")
    with pytest.raises(InvalidInput):
        await engine.remember("default", "   ")
    with pytest.raises(InvalidInput):
        await engine.remember("default", "x", created_at="yesterday-ish")
    with pytest.raises(InvalidInput):
        await engine.recall("default", "")


async def test_recall_reports_bytes_left_behind(engine):
    await engine.remember("default", "a" * 400)
    await engine.remember("default", "the only note about zebras")
    result = await engine.recall("default", "zebras", limit=1)
    assert result.returned_bytes == len("the only note about zebras")
    assert result.space_bytes == 400 + len("the only note about zebras")
    assert 0.9 < result.context_reduction < 1.0


async def test_vector_index_rejects_a_width_change(engine):
    with pytest.raises(ValueError):
        await engine.vectors.ensure(engine.embedder.dim + 1)


async def test_status_names_the_parts(engine):
    status = await engine.status("default")
    assert status.embedder == engine.embedder.id
    assert status.document_store == engine.documents.name
    assert status.vector_index == engine.vectors.name
    assert status.revision == 0
    await engine.remember("default", "one")
    assert (await engine.status("default")).revision == 1


async def test_where_scopes_recall_inside_a_space(engine):
    alice = await engine.remember(
        "default", "alice prefers the window seat on flights", metadata={"user_id": "alice", "agent_id": "travel"}
    )
    await engine.remember(
        "default", "bob prefers the aisle seat on flights", metadata={"user_id": "bob", "agent_id": "travel"}
    )
    result = await engine.recall("default", "seat preference on flights", where={"user_id": "alice"})
    assert [i.episode_id for i in result.items] == [alice.episode_id]
    assert result.items[0].metadata == {"user_id": "alice", "agent_id": "travel"}
    both = await engine.recall("default", "seat preference on flights", where={"agent_id": "travel"})
    assert len(both.items) == 2
    assert (await engine.recall("default", "seat preference", where={"user_id": "carol"})).items == []


async def test_metadata_is_bounded(engine):
    with pytest.raises(InvalidInput):
        await engine.remember("default", "x", metadata={"User-Id": "alice"})
    with pytest.raises(InvalidInput):
        await engine.remember("default", "x", metadata={"k": ""})
    with pytest.raises(InvalidInput):
        await engine.remember("default", "x", metadata={f"k{i}": "v" for i in range(17)})

async def test_new_object_closes_the_old_fact_with_a_reason(engine):
    old = await engine.assert_fact("default", "Mark", "lives_in", "Austin", valid_from="2022-01-01")
    new = await engine.assert_fact("default", "Mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    [closed] = [f for f in await engine.facts("default", include_closed=True) if f.fact_id == old.fact_id]
    assert closed.status == "closed"
    assert closed.valid_until == new.valid_from
    assert closed.closed_reason == f"superseded by fact {new.fact_id}"
    assert [f.object for f in await engine.facts("default")] == ["Lisbon"]
    assert [f.object for f in await engine.facts("default", as_of="2023-06-01")] == ["Austin"]


async def test_restating_a_fact_is_idempotent(engine):
    first = await engine.assert_fact("default", "mark", "drinks", "black coffee")
    again = await engine.assert_fact("default", "Mark ", "Drinks", "black coffee")
    assert again.fact_id == first.fact_id
    assert len(await engine.facts("default", include_closed=True)) == 1


async def test_a_late_arriving_stale_fact_does_not_overwrite_the_fresh_one(engine):
    fresh = await engine.assert_fact("default", "mark", "works_at", "Acme", valid_from="2024-06-01")
    stale = await engine.assert_fact("default", "mark", "works_at", "Initech", valid_from="2021-01-01")
    assert stale.status == "closed"
    assert stale.valid_until == fresh.valid_from
    assert stale.closed_reason == f"superseded by fact {fresh.fact_id}"
    active = await engine.facts("default")
    assert [f.object for f in active] == ["Acme"]
    assert [f.object for f in await engine.facts("default", as_of="2022-01-01")] == ["Initech"]


async def test_close_fact_keeps_the_reason(engine):
    fact = await engine.assert_fact("default", "mark", "allergic_to", "peanuts")
    engine.test_clock.now = "2025-02-01T00:00:00.000Z"
    closed = await engine.close_fact("default", fact.fact_id, "was a misdiagnosis")
    assert closed.status == "closed"
    assert closed.valid_until == "2025-02-01T00:00:00.000Z"
    assert closed.closed_reason == "was a misdiagnosis"
    assert await engine.facts("default") == []
    with pytest.raises(NotFound):
        await engine.close_fact("default", 999, "nope")
    with pytest.raises(InvalidInput):
        await engine.close_fact("default", fact.fact_id, "x" * 501)


async def test_recall_returns_facts_that_hold_at_the_time_asked(engine):
    await engine.assert_fact("default", "mark", "lives_in", "Austin", valid_from="2022-01-01")
    await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    await engine.remember("default", "unrelated note about gardening")
    now = await engine.recall("default", "where does mark live")
    assert [f.object for f in now.facts] == ["Lisbon"]
    then = await engine.recall("default", "where does mark live", as_of="2023-01-01")
    assert [f.object for f in then.facts] == ["Austin"]


async def test_profile_is_identity_plus_recent_activity(engine):
    await engine.assert_fact("default", "mark", "name", "ProjectScone contributors")
    await engine.remember("default", "older note", created_at="2024-01-01")
    await engine.remember("default", "newer note", created_at="2024-02-01")
    profile = await engine.profile("default")
    assert [f.object for f in profile.static_facts] == ["ProjectScone contributors"]
    assert profile.dynamic == ["newer note", "older note"]


__all__ = [
    "test_remember_then_recall_finds_it",
    "test_same_content_is_deduplicated",
    "test_spaces_do_not_leak",
    "test_as_of_hides_what_happened_later",
    "test_tag_filter_requires_every_tag",
    "test_a_broken_vector_lane_is_reported_not_fatal",
    "test_at_most_two_chunks_per_episode",
    "test_forget_removes_from_recall",
    "test_bad_input_is_refused_loudly",
    "test_recall_reports_bytes_left_behind",
    "test_vector_index_rejects_a_width_change",
    "test_status_names_the_parts",
    "test_where_scopes_recall_inside_a_space",
    "test_metadata_is_bounded",
    "test_new_object_closes_the_old_fact_with_a_reason",
    "test_restating_a_fact_is_idempotent",
    "test_a_late_arriving_stale_fact_does_not_overwrite_the_fresh_one",
    "test_close_fact_keeps_the_reason",
    "test_recall_returns_facts_that_hold_at_the_time_asked",
    "test_profile_is_identity_plus_recent_activity",
]
