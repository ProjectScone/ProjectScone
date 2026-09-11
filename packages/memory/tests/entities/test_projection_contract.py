"""Every document store gives the same entity projection for the same ledger.

One scripted ledger (a quoted remember, stated and dated facts, a proposal
approved and one left pending, a close, an exclusion, another space) is
written to each store and read back through its pager. Stores number facts
and episodes their own way, so the signature renumbers them by creation
order; everything else, entity ids included, must match the in-memory
reference in every status mode.
"""
from __future__ import annotations

import json

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.timeutil import parse_rfc3339
from scone_memory.entities.project import project_entities
from scone_memory.entities.read import read_ledger
from scone_memory.entities.view import counts
from scone_memory.testing import Clock

WHEN = parse_rfc3339("2025-06-01T00:00:00Z")
DAY = "2024-01-01T00:00:00Z"


async def script(engine: MemoryEngine) -> None:
    note = await engine.remember("alpha", "Dr. Alice Chen joined Acme Robotics in May 2021. Acme Robotics is in Lisbon.")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", source_episode_id=note.episode_id,
                             quote="Dr. Alice Chen joined Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "alice chen", "joined_on", "May 2021", valid_from=DAY)
    proposal = await engine.assert_fact("alpha", "alice chen", "knows", "Bob", proposed=True, valid_from=DAY)
    await engine.approve("alpha", proposal.fact_id)
    await engine.assert_fact("alpha", "carol", "knows", "Bob", proposed=True, valid_from=DAY)
    old = await engine.assert_fact("alpha", "alice chen", "lived_in", "Porto", valid_from="2020-01-01T00:00:00Z")
    await engine.close_fact("alpha", old.fact_id, "moved")
    hidden = await engine.assert_fact("alpha", "alice chen", "met", "Mallory", valid_from=DAY)
    await engine.exclude("alpha", hidden.fact_id, "private")
    await engine.assert_fact("beta", "zed", "works_at", "Globex", valid_from=DAY)


async def signature(engine: MemoryEngine) -> dict[str, object]:
    ledger = await read_ledger(engine, "alpha")
    facts = ledger.facts
    ordinal = {fact_id: number for number, fact_id in enumerate(sorted(fact.fact_id for fact in facts), 1)}
    sources = sorted({fact.source_episode_id for fact in facts if fact.source_episode_id is not None})
    episode = {episode_id: number for number, episode_id in enumerate(sources, 1)}
    renumbered = [fact.model_copy(update={
        "fact_id": ordinal[fact.fact_id],
        "source_episode_id": None if fact.source_episode_id is None else episode[fact.source_episode_id],
        "superseded_by": None if fact.superseded_by is None else ordinal.get(fact.superseded_by, -1)})
        for fact in facts]
    views = {}
    for mode in ("all", "current", "history", "proposed"):
        counted = [fact for fact in renumbered
                   if counts(fact.status, fact.excluded, fact.valid_from, fact.valid_until, mode, WHEN)]
        projection = project_entities("alpha", counted, revision=0)
        views[mode] = {"digest": projection.digest, "entities": sorted(entity.entity_id for entity in projection.entities),
                       "relations": len(projection.relations), "attributes": len(projection.attributes)}
    return {"facts": len(facts), "read_mode": ledger.read_mode, "reasons": list(ledger.reasons), "views": views}


async def test_every_store_projects_the_scripted_ledger_like_the_reference(ledger_engine):
    reference = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await script(reference)
    await script(ledger_engine)
    expected, found = await signature(reference), await signature(ledger_engine)
    assert found == expected, json.dumps({"expected": expected, "found": found}, indent=1)
    assert expected["views"]["current"]["relations"] >= 2 and expected["views"]["proposed"]["relations"] == 1
