"""The derivation pass (G05): claims that follow from the claims a space
already holds are proposed with their premises as links, never quoted,
never restated, validated the way extraction output is, and recorded as
a `derive` event. An unchanged group is not sent to the model twice."""

from __future__ import annotations

import json

import pytest

from scone_memory import FakeChat
from scone_memory.core.models import Fact
from scone_memory.ingestion.derive import Deriver
from scone_memory.ingestion.distill import DistillError
from scone_memory.memory.engine import derivation_groups


def reply(*entries):
    return json.dumps(list(entries))


def test_groups_join_claims_through_shared_names():
    def fact(i, s, p, o):
        return Fact(fact_id=i, space="default", subject=s, predicate=p, object=o, valid_from="2024-01-01T00:00:00Z")

    groups = derivation_groups([
        fact(1, "mark", "works_at", "acme"), fact(2, "acme", "based_in", "lisbon"), fact(3, "juniper", "is", "a cat"),
    ])
    assert sorted(sorted(f.fact_id for f in g) for g in groups) == [[1, 2], [3]], "mark and lisbon meet through acme"


async def test_a_derivation_is_proposed_with_its_premises_and_never_restated(engine):
    works = await engine.assert_fact("default", "mark", "works_at", "acme")
    based = await engine.assert_fact("default", "acme", "based_in", "lisbon")
    inference = {"subject": "mark", "predicate": "works_in", "object": "lisbon",
                 "premises": [works.fact_id, based.fact_id], "confidence": 0.7}
    chat = FakeChat([reply(inference), reply(inference)])
    deriver = Deriver(engine, chat)
    assert await engine.pending_derivation("default") == 1, "one group: the two claims meet through acme"

    first = await deriver.derive("default")
    assert (first.groups, first.sent, len(first.proposed), first.restated) == (1, 1, 1, 0)
    fact = first.proposed[0]
    assert (fact.subject, fact.predicate, fact.object, fact.status, fact.origin, fact.confidence) == (
        "mark", "works_in", "lisbon", "proposed", "inferred", 0.7)
    assert fact.source_episode_id is None and fact.quote is None, "no episode says it; the links are the provenance"
    premises = {l.to_fact for l in await engine.fact_links("default", fact.fact_id) if l.kind == "derived_from" and l.from_fact == fact.fact_id}
    assert premises == {works.fact_id, based.fact_id}
    assert str(works.fact_id) in chat.calls[0][1] and "acme" in chat.calls[0][1], "the group is sent with its ids"
    assert await engine.pending_derivation("default") == 0, "the group was seen at this membership"
    assert (await deriver.derive("default")).sent == 0, "an unchanged group is not sent again"

    await engine.assert_fact("default", "acme", "founded_in", "2010")
    assert await engine.pending_derivation("default") == 1, "a new claim in the group makes it eligible again"
    second = await deriver.derive("default")
    assert (second.sent, len(second.proposed), second.restated) == (1, 0, 1), "the same inference again is restated, not duplicated"
    assert [f.fact_id for f in await engine.facts("default", status="proposed")] == [fact.fact_id]
    assert len(await engine.fact_links("default", fact.fact_id)) == 2, "and no new links"
    events = await engine.events.query("default", kind="derive")
    assert sorted(e.payload["proposed"] for e in events) == [0, 0, 1], "every pass records its event, sent or not"
    assert all(e.payload["groups"] == 1 for e in events) and [e.payload["restated"] for e in events][0] == 1


async def test_derivations_are_validated_like_extractions(engine):
    works = await engine.assert_fact("default", "mark", "works_at", "acme")
    based = await engine.assert_fact("default", "acme", "based_in", "lisbon")
    both = [works.fact_id, based.fact_id]
    entries = [
        {"subject": "mark", "predicate": "lives_in", "object": "lisbon", "premises": [works.fact_id, 999]},
        {"subject": "mark", "predicate": "lives_in", "object": "lisbon", "premises": [works.fact_id]},
        {"subject": "Mark", "predicate": "works_at", "object": "ACME", "premises": both},
        {"subject": "", "predicate": "x", "object": "y", "premises": both},
        "not an object",
        {"subject": "mark", "predicate": "commutes", "object": "daily", "premises": [works.fact_id],
         "rule": "people commute to where they work"},
    ]
    outcome = await Deriver(engine, FakeChat([reply(*entries)])).derive("default")
    assert [f.predicate for f in outcome.proposed] == ["commutes"], "one premise plus a stated rule is enough"
    assert sorted(r.reason for r in outcome.rejected) == sorted(
        ["unknown_premise", "too_few_premises", "restates_premise", "malformed", "malformed"])
    assert outcome.as_payload()["rejected_reasons"] == {"malformed": 2, "restates_premise": 1, "too_few_premises": 1, "unknown_premise": 1}

    await engine.assert_fact("other", "a", "is", "b")
    await engine.assert_fact("other", "b", "is", "c")
    with pytest.raises(DistillError):
        await Deriver(engine, FakeChat(["no json here"])).derive("other")
