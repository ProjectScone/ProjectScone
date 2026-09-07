"""Faithful beliefs: origin, review of proposals, and exclusion.

Runs over the same ``engine`` fixture as ``contract``. Import with
``from scone_memory.testing.contract_beliefs import *``.
"""

from __future__ import annotations

import pytest

from ..core.errors import InvalidInput, NotFound


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


async def test_a_quoted_claim_is_checked_against_its_source(engine):
    """A quote is the exact text the claim rests on. It must be in the
    source episode; a claim with a source but no quote is stored and reads
    as ungrounded; a quote with no source has nothing to be checked against."""
    ep = await engine.remember("default", "Ana moved to Lisbon in March 2024 for the harbour job.", created_at="2024-03-02")
    grounded = await engine.assert_fact("default", "ana", "lives_in", "Lisbon", valid_from="2024-03-02", origin="extracted",
                                        proposed=True, source_episode_id=ep.episode_id, quote="Ana moved to Lisbon in March 2024")
    assert grounded.quote == "Ana moved to Lisbon in March 2024" and grounded.grounded is True
    with pytest.raises(InvalidInput, match="not a substring"):
        await engine.assert_fact("default", "ana", "works_at", "the harbour", origin="extracted", proposed=True,
                                 source_episode_id=ep.episode_id, quote="Ana works at the harbour")
    with pytest.raises(InvalidInput, match="empty"):
        await engine.assert_fact("default", "ana", "works_at", "x", source_episode_id=ep.episode_id, quote="   ")
    with pytest.raises(InvalidInput, match="source_episode_id"):
        await engine.assert_fact("default", "ana", "works_at", "x", quote="Ana moved")
    with pytest.raises(NotFound):
        await engine.assert_fact("default", "ana", "works_at", "x", source_episode_id=999, quote="Ana moved")
    ungrounded = await engine.assert_fact("default", "ana", "prefers", "mornings", origin="extracted", proposed=True, source_episode_id=ep.episode_id)
    assert ungrounded.quote is None and ungrounded.grounded is False, "legacy or quote-less extractions read as ungrounded"
    stated = await engine.assert_fact("default", "ana", "name", "Ana")
    assert stated.grounded is None, "a stated claim with no source is neither grounded nor ungrounded"
    assert [f.fact_id for f in await engine.facts("default", status="proposed")] == [grounded.fact_id, ungrounded.fact_id]
    # the quote survives approval and export
    await engine.approve("default", grounded.fact_id)
    dump = [r async for r in engine.export("default")]
    assert next(r for r in dump if r.get("type") == "fact" and r["object"] == "Lisbon")["quote"] == "Ana moved to Lisbon in March 2024"
    await engine.import_records("other", dump)
    moved = next(f for f in await engine.facts("other") if f.object == "Lisbon")
    assert moved.quote == "Ana moved to Lisbon in March 2024"


async def test_history_returns_the_chain_behind_a_matched_claim(engine):
    """Research experiment 3: a knowledge-update question needs what was
    believed before as well as what holds now. History is opt-in, oldest
    first, and made only of closed ledger facts on the same subject and
    predicate: not proposals, not excluded facts, not another subject's
    closed facts."""
    await engine.assert_fact("default", "mark", "lives_in", "Austin", valid_from="2019-08-01")
    await engine.assert_fact("default", "mark", "lives_in", "Berlin", valid_from="2022-01-01")
    lisbon = await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    await engine.assert_fact("default", "ana", "lives_in", "Rome", valid_from="2020-01-01")  # another subject's chain stays out
    await engine.assert_fact("default", "ana", "lives_in", "Oslo", valid_from="2021-01-01")
    hidden = await engine.assert_fact("default", "mark", "lives_in", "Nowhere", valid_from="2018-01-01")  # closed, then excluded
    await engine.exclude("default", hidden.fact_id, "a joke")
    await engine.assert_fact("default", "mark", "lives_in", "Porto", valid_from="2025-01-01", origin="extracted", proposed=True)

    plain = await engine.recall("default", "where does mark live")
    assert [f.object for f in plain.facts] == ["Lisbon"] and plain.history == []

    with_history = await engine.recall("default", "where does mark live", history=True)
    assert [f.object for f in with_history.facts] == ["Lisbon"]
    chain = with_history.history
    assert [f.object for f in chain] == ["Austin", "Berlin"], "oldest first; the excluded one and the proposal are not history"
    assert all(f.status == "closed" for f in chain)
    assert chain[0].valid_until == "2022-01-01T00:00:00.000Z" and chain[1].valid_until == lisbon.valid_from
    assert chain[1].closed_reason == f"superseded by fact {lisbon.fact_id}"

    await engine.assert_fact("default", "mark", "lives_in", "Madrid", valid_from="2025-06-01")  # closes Lisbon too
    then = await engine.recall("default", "where does mark live", as_of="2023-01-01", history=True)
    assert [f.object for f in then.facts] == ["Berlin"]
    assert [f.object for f in then.history] == ["Austin"], "only what came before the boundary; Lisbon is closed too but began after it"


__all__ = [
    "test_a_proposal_answers_nothing_until_a_person_approves_it",
    "test_a_proposal_is_not_touched_by_later_assertions",
    "test_a_declined_proposal_never_held_and_keeps_its_reason",
    "test_approving_a_duplicate_proposal_returns_the_held_fact",
    "test_excluding_hides_a_fact_without_rewriting_its_history",
    "test_origin_survives_export_and_import",
    "test_a_quoted_claim_is_checked_against_its_source",
    "test_history_returns_the_chain_behind_a_matched_claim",
    "test_an_extension_keeps_both_facts_and_links_them",
    "test_a_derivation_names_its_premises_and_is_labeled_inferred",
    "test_links_refuse_self_reference_unknown_kinds_and_dependency_cycles",
    "test_a_link_carries_its_evidence_and_outlives_a_forgotten_source",
    "test_a_proposed_derivation_is_linked_but_answers_nothing_until_approved",
]


async def test_an_extension_keeps_both_facts_and_links_them(engine):
    base = await engine.assert_fact("default", "mark", "lives_in", "Portugal", valid_from="2024-01-01")
    detail = await engine.assert_fact("default", "mark", "lives_in_city", "Lisbon", valid_from="2024-02-01", extends=base.fact_id)
    facts = {f.fact_id: f for f in await engine.facts("default")}
    assert facts[base.fact_id].status == "active" and facts[detail.fact_id].status == "active"
    links = await engine.fact_links("default", base.fact_id)
    assert [(link.from_fact, link.to_fact, link.kind) for link in links] == [(detail.fact_id, base.fact_id, "extends")]
    assert (await engine.fact_links("default", detail.fact_id)) == links, "a link reads the same from either end"
    # An extension cannot supersede what it extends: that is an update.
    with pytest.raises(InvalidInput):
        await engine.assert_fact("default", "mark", "lives_in", "Spain", valid_from="2025-01-01", extends=base.fact_id)
    assert (await engine.facts("default", include_closed=True)) == sorted(facts.values(), key=lambda f: f.fact_id), "a refused extension changes nothing"
    with pytest.raises(NotFound):
        await engine.assert_fact("default", "mark", "drinks", "tea", extends=999_999)
    elsewhere = await engine.assert_fact("other", "x", "y", "z")
    with pytest.raises(NotFound):
        await engine.assert_fact("default", "mark", "drinks", "tea", extends=elsewhere.fact_id)


async def test_a_derivation_names_its_premises_and_is_labeled_inferred(engine):
    works = await engine.assert_fact("default", "mark", "works_at", "Acme")
    based = await engine.assert_fact("default", "Acme", "based_in", "Lisbon")
    derived = await engine.assert_fact("default", "mark", "works_in", "Lisbon", derived_from=[works.fact_id, based.fact_id])
    assert derived.origin == "inferred"
    links = await engine.fact_links("default", derived.fact_id)
    assert sorted((l.from_fact, l.to_fact, l.kind) for l in links) == sorted(
        [(derived.fact_id, works.fact_id, "derived_from"), (derived.fact_id, based.fact_id, "derived_from")])
    again = await engine.assert_fact("default", "mark", "works_in", "Lisbon", derived_from=[works.fact_id, based.fact_id])
    assert again.fact_id == derived.fact_id and len(await engine.fact_links("default", derived.fact_id)) == 2, "restated, not relinked"
    # A proposal is not truth yet, so nothing can be derived from it.
    parked = await engine.assert_fact("default", "mark", "likes", "tea", proposed=True)
    with pytest.raises(InvalidInput):
        await engine.assert_fact("default", "mark", "drinks", "tea", derived_from=[parked.fact_id])
    # A derivation is inferred by definition; claiming it was extracted is refused.
    with pytest.raises(InvalidInput):
        await engine.assert_fact("default", "mark", "commutes_to", "Lisbon", derived_from=[works.fact_id], origin="extracted")
    with pytest.raises(NotFound):
        await engine.assert_fact("default", "mark", "commutes_to", "Lisbon", derived_from=[999_999])


async def test_links_refuse_self_reference_unknown_kinds_and_dependency_cycles(engine):
    a = await engine.assert_fact("default", "a", "is", "1")
    b = await engine.assert_fact("default", "b", "is", "2")
    c = await engine.assert_fact("default", "c", "is", "3")
    link = await engine.link_facts("default", a.fact_id, b.fact_id, "supports")
    assert (link.from_fact, link.to_fact, link.kind) == (a.fact_id, b.fact_id, "supports")
    settled = await engine.revision("default")
    assert (await engine.link_facts("default", a.fact_id, b.fact_id, "supports")).link_id == link.link_id, "the same link twice is one link"
    assert await engine.revision("default") == settled, "and it moves nothing"
    with pytest.raises(InvalidInput):
        await engine.link_facts("default", a.fact_id, a.fact_id, "supports")
    with pytest.raises(InvalidInput):
        await engine.link_facts("default", a.fact_id, b.fact_id, "resembles")
    await engine.link_facts("default", b.fact_id, c.fact_id, "derived_from")
    await engine.link_facts("default", c.fact_id, a.fact_id, "derived_from")
    with pytest.raises(InvalidInput):
        await engine.link_facts("default", a.fact_id, b.fact_id, "derived_from")  # a -> b -> c -> a
    with pytest.raises(InvalidInput):
        await engine.link_facts("default", a.fact_id, b.fact_id, "extends")  # extension is a dependency too
    await engine.link_facts("default", a.fact_id, b.fact_id, "contradicts")  # not a dependency, so no cycle
    elsewhere = await engine.assert_fact("other", "x", "y", "z")
    with pytest.raises(NotFound):
        await engine.link_facts("default", a.fact_id, elsewhere.fact_id, "supports")


async def test_a_link_carries_its_evidence_and_outlives_a_forgotten_source(engine):
    source = await engine.remember("default", "Acme is headquartered in Lisbon, near the river.")
    a = await engine.assert_fact("default", "mark", "works_at", "Acme")
    b = await engine.assert_fact("default", "Acme", "based_in", "Lisbon")
    with pytest.raises(InvalidInput):
        await engine.link_facts("default", a.fact_id, b.fact_id, "supports", source_episode_id=source.episode_id, quote="in Porto")
    link = await engine.link_facts("default", a.fact_id, b.fact_id, "supports",
                                   source_episode_id=source.episode_id, quote="headquartered in Lisbon")
    assert (link.source_episode_id, link.quote) == (source.episode_id, "headquartered in Lisbon")
    await engine.forget("default", source.episode_id)
    [kept] = await engine.fact_links("default", a.fact_id)
    assert kept.source_episode_id == source.episode_id, "the link records what supported it; its source being gone is the audit's business"


async def test_a_proposed_derivation_is_linked_but_answers_nothing_until_approved(engine):
    """Inference is not approved truth. A derived claim parked for review
    keeps its premises on record from the start, holds nothing while
    proposed, and never held if declined; approval is what admits it."""
    works = await engine.assert_fact("default", "mark", "works_at", "Acme")
    based = await engine.assert_fact("default", "Acme", "based_in", "Lisbon")
    guess = await engine.assert_fact("default", "mark", "works_in", "Lisbon", derived_from=[works.fact_id, based.fact_id], proposed=True)
    assert guess.status == "proposed" and guess.origin == "inferred"
    assert [l.kind for l in await engine.fact_links("default", guess.fact_id)] == ["derived_from", "derived_from"]
    assert guess.fact_id not in {f.fact_id for f in await engine.facts("default")}
    assert not guess.holds_at("2030-01-01T00:00:00Z"), "premises being held proves nothing about the inference"
    declined = await engine.decline("default", guess.fact_id, "does not follow")
    assert declined.status == "declined" and len(await engine.fact_links("default", guess.fact_id)) == 2, "the record of the attempt stays"
    with pytest.raises(InvalidInput):
        await engine.assert_fact("default", "mark", "commutes_to", "Lisbon", derived_from=[guess.fact_id])
    second = await engine.assert_fact("default", "mark", "lives_near", "Acme", derived_from=[works.fact_id], proposed=True)
    approved = await engine.approve("default", second.fact_id)
    assert approved.status == "active" and approved.fact_id in {f.fact_id for f in await engine.facts("default")}
    assert [l.to_fact for l in await engine.fact_links("default", approved.fact_id)] == [works.fact_id]
