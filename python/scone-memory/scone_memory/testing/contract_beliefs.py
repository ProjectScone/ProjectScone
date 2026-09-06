"""Faithful beliefs: origin, review of proposals, and exclusion.

Runs over the same ``engine`` fixture as ``contract``. Import with
``from scone_memory.testing.contract_beliefs import *``.
"""

from __future__ import annotations

import pytest

from ..errors import InvalidInput, NotFound


async def test_a_proposal_answers_nothing_until_a_person_approves_it(engine):
    austin = await engine.assert_fact("default", "mark", "lives_in", "Austin", valid_from="2022-01-01")
    lisbon = await engine.assert_fact(
        "default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02", origin="extracted", proposed=True, confidence=0.6
    )
    assert (lisbon.status, lisbon.origin) == ("proposed", "extracted")
    assert [f.object for f in await engine.facts("default")] == ["Austin"]
    assert [f.object for f in await engine.facts("default", as_of="2025-01-01")] == ["Austin"]
    assert [f.object for f in (await engine.recall("default", "mark lives")).facts] == ["Austin"]
    assert [f.fact_id for f in await engine.facts("default", status="proposed")] == [lisbon.fact_id]
    assert (await engine.status("default")).pending_review == 1

    approved = await engine.approve("default", lisbon.fact_id)
    assert (approved.status, approved.origin) == ("active", "extracted")
    assert [f.object for f in await engine.facts("default")] == ["Lisbon"]
    closed = await engine.documents.get_fact("default", austin.fact_id)
    assert (closed.status, closed.valid_until, closed.closed_reason) == ("closed", lisbon.valid_from, f"superseded by fact {lisbon.fact_id}")
    assert (await engine.status("default")).pending_review == 0
    with pytest.raises(InvalidInput):
        await engine.approve("default", lisbon.fact_id)  # no longer proposed


async def test_a_proposal_is_not_touched_by_later_assertions(engine):
    """Until approved, a proposal is outside the partition: it neither
    bounds a new fact nor gets truncated by one."""
    austin = await engine.assert_fact("default", "mark", "lives_in", "Austin", valid_from="2022-01-01")
    proposal = await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02", origin="extracted", proposed=True)
    berlin = await engine.assert_fact("default", "mark", "lives_in", "Berlin", valid_from="2025-01-01")
    untouched = await engine.documents.get_fact("default", proposal.fact_id)
    assert (untouched.status, untouched.valid_until, untouched.closed_reason) == ("proposed", None, None)
    assert (await engine.documents.get_fact("default", berlin.fact_id)).status == "active"
    assert (await engine.documents.get_fact("default", austin.fact_id)).valid_until == berlin.valid_from
    # Approval places it between the two: Austin ends earlier, Lisbon is bounded by Berlin.
    approved = await engine.approve("default", proposal.fact_id)
    assert (approved.status, approved.valid_until) == ("closed", berlin.valid_from)
    assert (await engine.documents.get_fact("default", austin.fact_id)).valid_until == proposal.valid_from
    assert [f.object for f in await engine.facts("default", as_of="2024-06-01")] == ["Lisbon"]


async def test_a_declined_proposal_never_held_and_keeps_its_reason(engine):
    proposal = await engine.assert_fact("default", "mark", "allergic_to", "peanuts", origin="extracted", proposed=True)
    with pytest.raises(InvalidInput):
        await engine.close_fact("default", proposal.fact_id, "not the right operation")
    declined = await engine.decline("default", proposal.fact_id, "the episode was about someone else")
    assert (declined.status, declined.closed_reason) == ("declined", "the episode was about someone else")
    assert await engine.facts("default") == []
    assert await engine.facts("default", as_of="2030-01-01") == []
    assert [f.fact_id for f in await engine.facts("default", status="declined")] == [proposal.fact_id]
    with pytest.raises(InvalidInput):
        await engine.decline("default", proposal.fact_id, "twice")
    with pytest.raises(NotFound):
        await engine.approve("default", 999)


async def test_approving_a_duplicate_proposal_returns_the_held_fact(engine):
    held = await engine.assert_fact("default", "mark", "drinks", "black coffee", valid_from="2024-01-01")
    proposal = await engine.assert_fact("default", "Mark", "drinks", "black coffee", valid_from="2024-06-01", origin="extracted", proposed=True)
    result = await engine.approve("default", proposal.fact_id)
    assert result.fact_id == held.fact_id
    stored = await engine.documents.get_fact("default", proposal.fact_id)
    assert (stored.status, stored.closed_reason) == ("declined", f"duplicate of fact {held.fact_id}")
    assert [f.fact_id for f in await engine.facts("default")] == [held.fact_id]


async def test_excluding_hides_a_fact_without_rewriting_its_history(engine):
    fact = await engine.assert_fact("default", "mark", "lives_in", "Austin", valid_from="2022-01-01")
    excluded = await engine.exclude("default", fact.fact_id, "shared by mistake")
    assert (excluded.status, excluded.valid_until, excluded.excluded_reason) == ("active", None, "shared by mistake")
    assert await engine.facts("default") == []
    assert (await engine.recall("default", "where does mark live")).facts == []
    assert [f.fact_id for f in await engine.facts("default", include_excluded=True)] == [fact.fact_id]
    assert (await engine.profile("default")).static_facts == []

    # History is not falsified: a later fact still truncates the excluded one.
    lisbon = await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    truncated = await engine.documents.get_fact("default", fact.fact_id)
    assert (truncated.valid_until, truncated.excluded_reason) == (lisbon.valid_from, "shared by mistake")

    included = await engine.include("default", fact.fact_id)
    assert included.excluded_reason is None
    assert [f.object for f in await engine.facts("default", as_of="2023-01-01")] == ["Austin"]
    with pytest.raises(InvalidInput):
        proposal = await engine.assert_fact("default", "x", "y", "z", proposed=True)
        await engine.exclude("default", proposal.fact_id, "not in the ledger")


async def test_origin_survives_export_and_import(engine):
    await engine.assert_fact("default", "mark", "uses", "neovim", origin="extracted", confidence=0.9)
    await engine.assert_fact("default", "mark", "prefers", "dark mode", origin="inferred", confidence=0.5)
    stated = await engine.assert_fact("default", "mark", "name", "Mark")
    await engine.exclude("default", stated.fact_id, "keep private")
    dump = [r async for r in engine.export("default")]
    summary = await engine.import_records("other", dump)
    assert summary.facts == 3
    moved = {f.object: f for f in await engine.facts("other", include_closed=True, include_excluded=True)}
    assert (moved["neovim"].origin, moved["dark mode"].origin, moved["Mark"].origin) == ("extracted", "inferred", "stated")
    assert moved["Mark"].excluded_reason == "keep private"
    with pytest.raises(InvalidInput):
        await engine.assert_fact("default", "a", "b", "c", origin="rumour")


__all__ = [
    "test_a_proposal_answers_nothing_until_a_person_approves_it",
    "test_a_proposal_is_not_touched_by_later_assertions",
    "test_a_declined_proposal_never_held_and_keeps_its_reason",
    "test_approving_a_duplicate_proposal_returns_the_held_fact",
    "test_excluding_hides_a_fact_without_rewriting_its_history",
    "test_origin_survives_export_and_import",
]
