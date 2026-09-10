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


REFUSAL = "I can't provide information or guidance on illegal or harmful activities."


async def two_groups(engine):
    """Two groups that share nothing, so each is sent to the model once."""
    await engine.assert_fact("default", "mark", "works_at", "acme")
    await engine.assert_fact("default", "acme", "based_in", "lisbon")
    await engine.assert_fact("default", "juniper", "sleeps_on", "the sofa")
    await engine.assert_fact("default", "the sofa", "is_in", "the study")


async def test_a_model_that_refuses_one_group_does_not_end_the_pass(engine):
    """A real corpus contains something a model will not answer about.
    Tonight a benchmark leg died on exactly this: one refusal, and every
    group after it went unread. What the model would not touch has to
    cost that group and nothing else."""
    await two_groups(engine)
    inference = {"subject": "juniper", "predicate": "sleeps_in", "object": "the study",
                 "premises": [3, 4], "confidence": 0.6}
    chat = FakeChat([REFUSAL, reply(inference)])

    outcome = await Deriver(engine, chat).derive("default")

    assert len(outcome.proposed) == 1, "the group the model would answer about was still derived"
    assert outcome.sent == 2, "both groups were tried"
    assert [r.reason for r in outcome.rejected] == ["unreadable_reply"]
    assert outcome.as_payload()["rejected_reasons"] == {"unreadable_reply": 1}


async def test_a_model_that_refuses_everything_is_a_failure_not_an_empty_answer(engine):
    """Nothing follows and nothing could be read are different results.
    Reporting the second as the first is how a pass that achieved nothing
    is recorded as a pass that found nothing to do."""
    await two_groups(engine)
    chat = FakeChat([REFUSAL, REFUSAL])

    with pytest.raises(DistillError, match="no group could be read"):
        await Deriver(engine, chat).derive("default")


async def test_a_group_the_model_never_answered_is_tried_again_next_pass(engine):
    """A refusal is settled and there is no point asking twice. A call
    that never arrived is not settled, and dropping it would lose the
    group silently until something else changed its membership."""
    from scone_memory.providers.llm import ChatError

    await two_groups(engine)
    inference = {"subject": "juniper", "predicate": "sleeps_in", "object": "the study",
                 "premises": [3, 4], "confidence": 0.6}
    chat = FakeChat([ChatError("chat server unreachable"), reply(inference)])
    first = await Deriver(engine, chat).derive("default")
    assert [r.reason for r in first.rejected] == ["unreachable"]

    again = FakeChat([reply({"subject": "mark", "predicate": "works_in", "object": "lisbon",
                             "premises": [1, 2], "confidence": 0.7})])
    second = await Deriver(engine, chat=again).derive("default")

    assert len(second.proposed) == 1, "the group that never got an answer was asked again"
    assert second.sent == 1, "and the group that did answer was not asked twice"
