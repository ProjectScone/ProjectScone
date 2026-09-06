import pytest

from scone_memory import InvalidInput, NotFound


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
