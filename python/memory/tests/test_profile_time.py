"""Profile validity uses instants, regardless of timestamp spelling."""
from __future__ import annotations

import pytest

from scone_memory import MemoryEngine
from scone_memory.core.ports import NewFact


@pytest.mark.parametrize("clock", [
    "2025-01-01T01:00:00+01:00",
    "2024-12-31T18:00:00-06:00",
    "2025-01-01T00:00:00Z",
])
async def test_profile_agrees_with_fact_validity_for_equivalent_clocks(
    engine: MemoryEngine, clock: str,
) -> None:
    engine.clock = lambda: clock
    past = await engine.assert_fact("alpha", "past", "value", "held", valid_from="2024-12-31T23:59:59.999Z")
    current = await engine.assert_fact("alpha", "current", "value", "held", valid_from="2025-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "future", "value", "waiting", valid_from="2025-01-01T00:00:00.001Z")

    profile = await engine.profile("alpha")
    assert [fact.fact_id for fact in profile.static_facts] == [past.fact_id, current.fact_id]
    assert profile.static_facts == await engine.facts("alpha", as_of=clock)


async def test_profile_preserves_exact_interval_boundaries_from_storage(engine: MemoryEngine) -> None:
    engine.clock = lambda: "2025-01-01T00:00:00.000Z"
    # Document-store ports can supply valid RFC 3339 offsets and microseconds.
    # The profile must honor them without rewriting the retained records.
    cases = [
        ("2024-12-31T19:00:00-05:00", None, True),
        ("2024-12-31T19:00:01-05:00", None, False),
        ("2025-01-01T01:00:00+02:00", None, True),
        ("2024-01-01", "2025-01-01T01:00:00+01:00", False),
        ("2024-01-01", "2025-01-01T00:00:00.000001Z", True),
        ("2025-01-01T00:00:00.000001Z", None, False),
    ]
    expected = []
    for index, (start, end, held) in enumerate(cases):
        fact = await engine.documents.insert_fact(NewFact("alpha", f"subject {index}", "value", "recorded",
            start, valid_until=end, status="active"))
        if held:
            expected.append(fact.fact_id)
    before = await engine.documents.list_facts("alpha", include_closed=True)

    profile = await engine.profile("alpha")
    assert [fact.fact_id for fact in profile.static_facts] == expected
    assert await engine.documents.list_facts("alpha", include_closed=True) == before
