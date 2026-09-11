"""A value stated again later is remembered, so the order facts arrive in
never decides what held (G04).

Stating a claim again from a later day, while it already holds, changes
nothing a reader sees: the ledger keeps one fact. But the later start is
kept as an affirmation, so when a backfill arrives afterwards and cuts
that fact short, the claim resumes from the day it was stated again
instead of being lost. Every arrival order of the same history leaves the
same partition of time.
"""

from __future__ import annotations

from itertools import permutations
import random

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.observability.events import InMemoryEventLog
from scone_memory.core.timeutil import parse_rfc3339
from scone_memory.testing import Clock


def day(year: int, month: int = 1) -> str:
    return f"{year}-{month:02d}-01T00:00:00Z"


async def holding(engine, at: str, subject: str = "alice", predicate: str = "works_at") -> list[str]:
    instant = parse_rfc3339(at)
    return [fact.object for fact in await engine.documents.facts_for("alpha", subject, predicate)
            if fact.in_ledger and parse_rfc3339(fact.valid_from) <= instant
            and (fact.valid_until is None or parse_rfc3339(fact.valid_until) > instant)]


async def test_a_restatement_keeps_when_it_was_said_and_nothing_more(ledger_engine):
    """The ledger still holds one fact; the space's history gained the day
    it was said again, so its revision moves once, and saying the same
    again moves nothing."""
    engine = ledger_engine
    first = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    revision = await engine.documents.revision("alpha")
    again = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), origin="extracted")
    assert await engine.documents.revision("alpha") == revision + 1
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    assert await engine.documents.revision("alpha") == revision + 1
    assert again.fact_id == first.fact_id and len(await engine.documents.list_facts("alpha", True)) == 1
    kept = await engine.documents.affirmations("alpha", first.fact_id)
    assert [(a.valid_from, a.origin) for a in kept] == [("2023-01-01T00:00:00.000Z", "extracted")]


async def test_a_late_backfill_does_not_erase_a_value_that_returned(ledger_engine):
    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), confidence=0.7)
    globex = await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    assert await holding(engine, day(2020, 6)) == ["Acme"]
    assert await holding(engine, day(2022, 6)) == ["Globex"]
    assert await holding(engine, day(2024, 6)) == ["Acme"]
    resumed = next(fact for fact in await engine.documents.facts_for("alpha", "alice", "works_at")
                   if fact.object == "Acme" and fact.valid_from.startswith("2023"))
    assert resumed.fact_id != acme.fact_id and resumed.confidence == 0.7 and resumed.valid_until is None
    stored = await engine.documents.get_fact("alpha", globex.fact_id)
    assert stored.valid_until == resumed.valid_from and stored.superseded_by == resumed.fact_id
    assert await engine.documents.affirmations("alpha", acme.fact_id) == [], "the affirmation became the fact"


async def test_a_resumed_value_keeps_how_the_first_one_ended_and_its_exclusion(ledger_engine):
    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    initech = await engine.assert_fact("alpha", "alice", "works_at", "Initech", valid_from=day(2025))
    await engine.exclude("alpha", acme.fact_id, "private")
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    resumed = next(fact for fact in await engine.documents.facts_for("alpha", "alice", "works_at")
                   if fact.object == "Acme" and fact.valid_from.startswith("2023"))
    assert resumed.valid_until == "2025-01-01T00:00:00.000Z" and resumed.status == "closed"
    assert resumed.superseded_by == initech.fact_id and resumed.excluded_reason == "private"
    assert await holding(engine, day(2026)) == ["Initech"]


async def test_later_affirmations_follow_the_resumed_value(ledger_engine):
    engine = ledger_engine
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2027))
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    await engine.assert_fact("alpha", "alice", "works_at", "Initech", valid_from=day(2024))
    assert [await holding(engine, day(year, 6)) for year in (2020, 2022, 2023, 2025, 2028)] == [
        ["Acme"], ["Globex"], ["Acme"], ["Initech"], ["Acme"]]


async def test_an_approved_proposal_resumes_a_returned_value_too(ledger_engine):
    engine = ledger_engine
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    proposal = await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021), proposed=True)
    await engine.approve("alpha", proposal.fact_id)
    assert [await holding(engine, day(year, 6)) for year in (2020, 2022, 2024)] == [["Acme"], ["Globex"], ["Acme"]]


async def test_an_approved_duplicate_proposal_is_kept_as_an_affirmation(ledger_engine):
    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    proposal = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), proposed=True)
    assert (await engine.approve("alpha", proposal.fact_id)).fact_id == acme.fact_id
    assert [a.valid_from for a in await engine.documents.affirmations("alpha", acme.fact_id)] == [
        "2023-01-01T00:00:00.000Z"]


async def test_deleting_a_space_removes_its_affirmations(ledger_engine):
    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    await engine.delete_space("alpha")
    assert await engine.documents.affirmations("alpha", acme.fact_id) == []


async def test_an_archive_carries_affirmations_to_a_new_store():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    records = [record async for record in source.export("alpha")]
    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await target.import_records("alpha", records)
    await target.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    assert await holding(target, day(2024, 6)) == ["Acme"]


HISTORY_VALUES = ("Acme", "Globex", "Initech")


@pytest.mark.parametrize("seed", range(60))
async def test_what_held_never_depends_on_the_order_facts_arrive_in(seed):
    """Random histories, each told in three random orders: what held at
    every sampled moment is the value of the row that began last by then."""
    rng = random.Random(seed)
    years = sorted(rng.sample(range(2000, 2030), k=rng.randint(3, 7)))
    history = [(year, rng.choice(HISTORY_VALUES)) for year in years]

    def expected(year: int, month: int) -> list[str]:
        started = [value for start, value in history if start < year or (start == year and month >= 1)]
        return started[-1:]

    orders = [list(order) for order in permutations(history)]
    for order in rng.sample(orders, k=min(3, len(orders))):
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                    clock=Clock()).open()
        for year, value in order:
            await engine.assert_fact("alpha", "alice", "works_at", value, valid_from=day(year))
        for year in range(1999, 2031):
            assert await holding(engine, day(year, 6)) == expected(year, 6), (history, order, year)


async def test_a_placement_that_fails_part_way_leaves_no_trace(ledger_engine, monkeypatch):
    """On a store that holds a transaction, the continuation, the new fact,
    the cut and the moved affirmations commit together or not at all."""
    engine, documents = ledger_engine, ledger_engine.documents
    if not callable(getattr(documents, "atomic", None)):
        pytest.skip(f"{type(documents).__name__} writes each fact alone")
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    before, revision = await documents.list_facts("alpha", True), await documents.revision("alpha")

    async def broken(fact):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(documents, "update_fact", broken)
    with pytest.raises(RuntimeError):
        await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    monkeypatch.undo()
    assert await documents.list_facts("alpha", True) == before and await documents.revision("alpha") == revision
    assert [a.valid_from for a in await documents.affirmations("alpha", acme.fact_id)] == ["2023-01-01T00:00:00.000Z"]


async def test_every_store_keeps_affirmations_by_fact_and_day(ledger_engine):
    """The port each store implements: one record per (fact, day) however
    often it is kept, earliest day first, dropped by id, listed by space."""
    from scone_memory.core.affirmations import NewAffirmation, affirmation_store

    engine = ledger_engine
    store = affirmation_store(engine.documents)
    assert store is not None, f"{type(engine.documents).__name__} keeps no affirmations"
    fact = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))

    def kept(year: int, **extra) -> NewAffirmation:
        return NewAffirmation(space="alpha", fact_id=fact.fact_id, valid_from=f"{year}-01-01T00:00:00.000Z",
                              recorded_at="2026-01-01T00:00:00.000Z", **extra)

    late = await store.add_affirmation(kept(2025, origin="extracted", quote=None, confidence=0.5))
    early = await store.add_affirmation(kept(2022))
    assert (await store.add_affirmation(kept(2025))).affirmation_id == late.affirmation_id
    listed = await store.affirmations("alpha", fact.fact_id)
    assert [(a.valid_from[:4], a.origin, a.confidence) for a in listed] == [("2022", "stated", 1.0),
                                                                           ("2025", "extracted", 0.5)]
    assert {a.affirmation_id for a in await store.space_affirmations("alpha")} == {early.affirmation_id,
                                                                                 late.affirmation_id}
    assert await store.affirmations("beta", fact.fact_id) == []
    await store.drop_affirmations("alpha", [early.affirmation_id])
    assert [a.affirmation_id for a in await store.affirmations("alpha", fact.fact_id)] == [late.affirmation_id]


async def intervals(engine) -> list[tuple[str, str, str | None]]:
    return sorted((fact.object, fact.valid_from[:4], fact.valid_until and fact.valid_until[:4])
                  for fact in await engine.documents.list_facts("alpha", True) if fact.in_ledger)


async def test_a_close_by_hand_and_a_later_restatement_agree_in_either_order():
    """A person closes Acme now (2025), and Acme is stated again from 2030.
    Told close-then-restate, 2030 is a new fact; told restate-then-close,
    the kept affirmation resumes there. A backfill after either leaves the
    same ledger, and no interval ends before it starts."""
    ledgers = []
    for restate_first in (True, False):
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                    clock=Clock(), events=InMemoryEventLog()).open()
        acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
        if restate_first:
            await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2030))
        closed = await engine.close_fact("alpha", acme.fact_id, "ended in 2025")
        if not restate_first:
            await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2030))
        assert closed.valid_until == "2025-01-01T00:00:00.000Z"
        if restate_first:
            [close] = await engine.events.query("alpha", kind="fact_close", limit=1)
            resumed = next(fact for fact in await engine.documents.list_facts("alpha", True)
                           if fact.valid_from.startswith("2030"))
            assert close.payload["resumed"] == resumed.fact_id
        await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
        for fact in await engine.documents.list_facts("alpha", True):
            assert fact.valid_until is None or parse_rfc3339(fact.valid_until) > parse_rfc3339(fact.valid_from), fact
        assert await engine.documents.affirmations("alpha", acme.fact_id) == []
        ledgers.append(await intervals(engine))
    assert ledgers[0] == ledgers[1] == [("Acme", "2020", "2021"), ("Acme", "2030", None), ("Globex", "2021", "2030")]


async def test_a_close_at_the_moment_of_a_restatement_ends_it_too(ledger_engine):
    """Said again now and closed now: the close is the later word, so the
    claim ends and nothing resumes."""
    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme")
    await engine.close_fact("alpha", acme.fact_id, "left")
    assert await intervals(engine) == [("Acme", "2020", "2025")]


async def test_a_fact_that_has_not_begun_cannot_be_closed_before_it_begins(ledger_engine):
    from scone_memory.core.errors import InvalidInput

    engine = ledger_engine
    planned = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2030))
    with pytest.raises(InvalidInput, match="begins"):
        await engine.close_fact("alpha", planned.fact_id, "never happened")
    assert (await engine.documents.get_fact("alpha", planned.fact_id)).status == "active"


@pytest.mark.parametrize("kept", ["2025-01-01T00:00:00.000Z", "2030-01-01T00:00:00.000Z"])
async def test_an_affirmation_at_or_past_a_facts_end_never_resumes(ledger_engine, kept):
    """However one came to lie at or past where its fact ended (an old
    archive, a store written by hand), it resumes nothing: resuming there
    would end the claim before, or as, it starts."""
    from scone_memory.core.affirmations import NewAffirmation, affirmation_store

    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.close_fact("alpha", acme.fact_id, "left")
    await affirmation_store(engine.documents).add_affirmation(NewAffirmation(
        space="alpha", fact_id=acme.fact_id, valid_from=kept, recorded_at="2025-01-01T00:00:00.000Z"))
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    assert await intervals(engine) == [("Acme", "2020", "2021"), ("Globex", "2021", None)]


async def premises(engine, fact_id: int) -> list[tuple[str, int]]:
    return sorted((link.kind, link.to_fact) for link in await engine.documents.fact_links("alpha", fact_id)
                  if link.from_fact == fact_id)


async def test_a_resumed_inference_keeps_the_premises_it_was_stated_with(ledger_engine):
    """Acme from 2020 inferred from one fact, from 2023 from another: the
    fact that resumes in 2023 rests on the second alone, as it would had
    the backfill come first."""
    engine = ledger_engine
    badge = await engine.assert_fact("alpha", "alice", "has_badge", "acme", valid_from=day(2019))
    payroll = await engine.assert_fact("alpha", "alice", "on_payroll_of", "Acme", valid_from=day(2019))
    title = await engine.assert_fact("alpha", "alice", "title", "engineer", valid_from=day(2019))
    first = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020),
                                     derived_from=[badge.fact_id])
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023),
                             derived_from=[payroll.fact_id], extends=title.fact_id)
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    resumed = next(fact for fact in await engine.documents.facts_for("alpha", "alice", "works_at")
                   if fact.in_ledger and fact.object == "Acme" and fact.valid_from.startswith("2023"))
    assert resumed.origin == "inferred"
    assert await premises(engine, resumed.fact_id) == [("derived_from", payroll.fact_id), ("extends", title.fact_id)]
    assert ("derived_from", badge.fact_id) in await premises(engine, first.fact_id)


async def test_later_affirmations_take_their_premises_with_them(ledger_engine):
    engine = ledger_engine
    payroll = await engine.assert_fact("alpha", "alice", "on_payroll_of", "Acme", valid_from=day(2019))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2027), derived_from=[payroll.fact_id])
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    await engine.assert_fact("alpha", "alice", "works_at", "Initech", valid_from=day(2024))
    resumed = next(fact for fact in await engine.documents.facts_for("alpha", "alice", "works_at")
                   if fact.in_ledger and fact.object == "Acme" and fact.valid_from.startswith("2027"))
    assert await premises(engine, resumed.fact_id) == [("derived_from", payroll.fact_id)]


async def test_an_approved_duplicate_keeps_the_proposals_premises_on_its_affirmation(ledger_engine):
    engine = ledger_engine
    payroll = await engine.assert_fact("alpha", "alice", "on_payroll_of", "Acme", valid_from=day(2019))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    proposal = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), proposed=True,
                                        derived_from=[payroll.fact_id])
    await engine.approve("alpha", proposal.fact_id)
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    resumed = next(fact for fact in await engine.documents.facts_for("alpha", "alice", "works_at")
                   if fact.in_ledger and fact.object == "Acme" and fact.valid_from.startswith("2023"))
    assert await premises(engine, resumed.fact_id) == [("derived_from", payroll.fact_id)]


async def test_an_archive_carries_an_affirmations_premises_by_their_new_ids():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    payroll = await source.assert_fact("alpha", "alice", "on_payroll_of", "Acme", valid_from=day(2019))
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), derived_from=[payroll.fact_id])
    records = [record async for record in source.export("alpha")]
    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await target.assert_fact("alpha", "bob", "works_at", "Initech", valid_from=day(2018))
    await target.import_records("alpha", records)
    await target.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    moved = next(fact for fact in await target.documents.facts_for("alpha", "alice", "on_payroll_of"))
    resumed = next(fact for fact in await target.documents.facts_for("alpha", "alice", "works_at")
                   if fact.in_ledger and fact.object == "Acme" and fact.valid_from.startswith("2023"))
    assert moved.fact_id != payroll.fact_id
    assert await premises(target, resumed.fact_id) == [("derived_from", moved.fact_id)]


async def test_an_import_that_adds_only_affirmations_moves_the_revision_once():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    first = [record async for record in source.export("alpha")]
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023))
    second = [record async for record in source.export("alpha")]
    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await target.import_records("alpha", first)
    before = await target.documents.revision("alpha")
    added = await target.import_records("alpha", second)
    assert (added.facts, added.affirmations, added.affirmations_skipped) == (0, 1, 0)
    assert await target.documents.revision("alpha") == before + 1
    again = await target.import_records("alpha", second)
    assert (again.affirmations, again.affirmations_skipped) == (0, 1)
    assert await target.documents.revision("alpha") == before + 1


async def test_forgetting_a_source_says_which_restatements_cite_it(ledger_engine):
    """Claims stand, restatements too; the receipt names both, before and
    after, so a restatement that later resumes as a fact is not a surprise."""
    engine = ledger_engine
    acme = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    said = await engine.remember("alpha", "Alice works at Acme in 2023")
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023),
                             source_episode_id=said.episode_id, quote="Alice works at Acme")
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2024))
    kept, unsourced = await engine.documents.affirmations("alpha", acme.fact_id)
    preview = await engine.impact("alpha", said.episode_id)
    receipt = await engine.forget("alpha", said.episode_id)
    assert preview.affirmations_citing == receipt.affirmations_citing == [kept.affirmation_id]
    assert (preview.facts_citing, preview.links_citing) == ([], [])
    assert [a.source_episode_id for a in await engine.documents.affirmations("alpha", acme.fact_id)] == [
        said.episode_id, None]


async def test_an_approved_duplicate_keeps_only_what_it_rests_on(ledger_engine):
    """A proposal's other links (what it supports, what rests on it) are
    about the proposal; only what it extends or was derived from is a
    premise."""
    engine = ledger_engine
    payroll = await engine.assert_fact("alpha", "alice", "on_payroll_of", "Acme", valid_from=day(2019))
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    proposal = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), proposed=True)
    await engine.link_facts("alpha", proposal.fact_id, payroll.fact_id, "supports")
    await engine.link_facts("alpha", payroll.fact_id, proposal.fact_id, "derived_from")
    await engine.approve("alpha", proposal.fact_id)
    await engine.assert_fact("alpha", "alice", "works_at", "Globex", valid_from=day(2021))
    resumed = next(fact for fact in await engine.documents.facts_for("alpha", "alice", "works_at")
                   if fact.in_ledger and fact.object == "Acme" and fact.valid_from.startswith("2023"))
    assert await premises(engine, resumed.fact_id) == []


async def test_an_archived_premise_that_cannot_be_named_is_dropped_and_counted():
    """An affirmation's premise missing from the archive, or of a kind no
    premise has, is left off it and counted with the dropped links; the
    affirmation itself still arrives."""
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    payroll = await source.assert_fact("alpha", "alice", "on_payroll_of", "Acme", valid_from=day(2019))
    acme = await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2020))
    await source.assert_fact("alpha", "alice", "works_at", "Acme", valid_from=day(2023), derived_from=[payroll.fact_id])
    records = []
    async for record in source.export("alpha"):
        if record["type"] == "fact" and record["fact_id"] == payroll.fact_id:
            continue
        if record["type"] == "fact_link" and payroll.fact_id in (record["from_fact"], record["to_fact"]):
            continue
        if record["type"] == "affirmation":
            record = {**record, "links": [*record["links"], ("supports", acme.fact_id)]}
        records.append(record)
    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    summary = await target.import_records("alpha", records)
    assert (summary.affirmations, summary.links_skipped) == (1, 2)
    [kept] = await target.documents.space_affirmations("alpha")
    assert kept.links == []
