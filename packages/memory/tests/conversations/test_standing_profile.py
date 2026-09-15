"""What the space holds about its subject reaches every turn, not only the
turns whose words happen to retrieve it.

Recall answers the question in front of it. A standing claim -- "the user
is vegetarian" -- is relevant to "what should I cook tonight" and shares
no word with it, so a conversation that only recalls never shows it to
the model. The profile already exists (the claims that hold now, most
often and most recently said first, each with its evidence, bounded and
one revision's worth); this puts it in front of each turn when a
conversation asks for it, bounded again by bytes, with the claim ids
named in the receipt. Proposed and excluded claims never appear: the
profile reads only claims that hold and are not excluded.
"""
from __future__ import annotations

import json

import pytest

from scone_memory.memory.catalog import ProfilePolicy
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation


class Model:
    def __init__(self):
        self.requests = []

    async def respond(self, request):
        self.requests.append(request)
        yield TextDelta("noted")
        yield ReplyCompleted()

    async def aclose(self):
        pass


def standing_block(request):
    found = [m["content"] for m in request if isinstance(m.get("content"), str)
             and m["content"].startswith("Scone standing claims")]
    assert len(found) <= 1
    return json.loads(found[0][found[0].index("{"):]) if found else None


async def seeded(engine):
    diet = await engine.assert_fact("alpha", "user", "diet", "vegetarian")
    city = await engine.assert_fact("alpha", "user", "lives_in", "Lisbon")
    proposed = await engine.assert_fact("alpha", "user", "allergic_to", "peanuts", proposed=True)
    excluded = await engine.assert_fact("alpha", "user", "prefers", "tea")
    await engine.exclude("alpha", excluded.fact_id, "a person said so")
    return diet, city, proposed, excluded


async def test_a_standing_claim_reaches_a_turn_that_would_not_retrieve_it(engine):
    diet, city, proposed, excluded = await seeded(engine)
    model = Model()
    conversation = TextConversation(engine, "alpha", "cook", lambda: model, standing_profile=ProfilePolicy())
    result = await conversation.reply("What should I make for dinner tonight?")
    block = standing_block(model.requests[-1])
    assert block is not None
    shown = {claim["fact_id"] for claim in block["claims"]}
    assert {diet.fact_id, city.fact_id} <= shown
    assert proposed.fact_id not in shown and excluded.fact_id not in shown
    context = result["memory_context"]
    assert set(context["profile_fact_ids"]) == shown and context["profile_status"] == "prepared"
    assert context["profile_truncated"] is False and context["profile_bytes"] > 0
    await conversation.close()


async def test_without_a_profile_policy_nothing_is_added(engine):
    await seeded(engine)
    model = Model()
    conversation = TextConversation(engine, "alpha", "plain", lambda: model)
    result = await conversation.reply("What should I make for dinner tonight?")
    assert standing_block(model.requests[-1]) is None
    assert "profile_status" not in result["memory_context"]
    await conversation.close()


async def test_the_block_is_bounded_and_says_what_it_left_out(engine):
    await seeded(engine)
    model = Model()
    conversation = TextConversation(engine, "alpha", "tight", lambda: model, standing_profile=ProfilePolicy(),
                                    max_profile_bytes=200)
    result = await conversation.reply("dinner?")
    context = result["memory_context"]
    block = standing_block(model.requests[-1])
    assert context["profile_truncated"] is True and context["profile_omitted_count"] >= 1
    assert len(json.dumps(block, separators=(",", ":")).encode()) <= 200
    assert block["coverage"]["omitted"] == context["profile_omitted_count"]
    await conversation.close()


async def test_the_policy_decides_which_predicates_stand(engine):
    diet, city, _, _ = await seeded(engine)
    model = Model()
    conversation = TextConversation(engine, "alpha", "policy", lambda: model,
                                    standing_profile=ProfilePolicy.of(predicates=["diet"]))
    await conversation.reply("dinner?")
    assert [claim["fact_id"] for claim in standing_block(model.requests[-1])["claims"]] == [diet.fact_id]
    await conversation.close()


async def test_a_profile_that_cannot_be_read_does_not_fail_the_turn(engine, monkeypatch):
    await seeded(engine)
    from scone_memory.memory import catalog

    async def broken(*args, **kwargs):
        raise RuntimeError("private detail")

    monkeypatch.setattr(catalog, "profile", broken)
    model = Model()
    conversation = TextConversation(engine, "alpha", "broken", lambda: model, standing_profile=ProfilePolicy())
    result = await conversation.reply("dinner?")
    assert result["text"] == "noted" and standing_block(model.requests[-1]) is None
    context = result["memory_context"]
    assert context["profile_status"] == "failed" and context["profile_fact_ids"] == []
    assert "private" not in json.dumps(context)
    await conversation.close()


async def test_a_bad_byte_bound_is_refused(engine):
    with pytest.raises(ValueError, match="max_profile_bytes"):
        TextConversation(engine, "alpha", "bad", lambda: Model(), standing_profile=ProfilePolicy(), max_profile_bytes=10)


async def dated(engine):
    """A settled claim from years ago and one from today, by the test clock."""
    diet = await engine.assert_fact("alpha", "user", "diet", "vegetarian", valid_from="2020-01-01T00:00:00Z")
    city = await engine.assert_fact("alpha", "user", "lives_in", "Lisbon", valid_from="2019-01-01T00:00:00Z")
    trip = await engine.assert_fact("alpha", "user", "travelling_to", "Kyoto")
    return diet, city, trip


@pytest.mark.parametrize("include, static, dynamic", [("static", True, False), ("dynamic", False, True),
                                                      ("both", True, True)])
async def test_a_conversation_chooses_which_buckets_stand(engine, include, static, dynamic):
    from scone_memory.memory.catalog import BucketBounds

    diet, city, trip = await dated(engine)
    model = Model()
    conversation = TextConversation(engine, "alpha", f"buckets-{include}", lambda: model,
                                    standing_profile=ProfilePolicy(), profile_buckets=BucketBounds(include=include))
    result = await conversation.reply("What should I make for dinner tonight?")
    block = standing_block(model.requests[-1])
    assert block["schema_version"] == 2 and ("static" in block, "dynamic" in block) == (static, dynamic)
    assert "claims" not in block
    context = result["memory_context"]
    expected = ([diet.fact_id, city.fact_id] if static else []) + ([trip.fact_id] if dynamic else [])
    assert sorted(context["profile_fact_ids"]) == sorted(expected) and context["profile_status"] == "prepared"
    if static:
        assert {claim["fact_id"] for claim in block["static"]} == {diet.fact_id, city.fact_id}
        assert {claim["rule"] for claim in block["static"]} == {"tenure"}
        assert context["profile_buckets"]["static"]["rules"] == {str(diet.fact_id): "tenure", str(city.fact_id): "tenure"}
    if dynamic:
        assert [claim["fact_id"] for claim in block["dynamic"]] == [trip.fact_id]
        assert context["profile_buckets"]["dynamic"]["fact_ids"] == [trip.fact_id]
    assert ("static" in context["profile_buckets"], "dynamic" in context["profile_buckets"]) == (static, dynamic)
    assert context["profile_bytes"] == len(model.requests[-1][
        next(i for i, m in enumerate(model.requests[-1]) if str(m.get("content", "")).startswith("Scone standing"))
    ]["content"].encode())
    await conversation.close()


async def test_each_standing_bucket_is_bounded_on_its_own_and_says_so(engine):
    from scone_memory.memory.catalog import BucketBounds

    await dated(engine)
    model = Model()
    conversation = TextConversation(engine, "alpha", "bucket-bounds", lambda: model, standing_profile=ProfilePolicy(),
                                    profile_buckets=BucketBounds(static_limit=1))
    result = await conversation.reply("dinner?")
    block, context = standing_block(model.requests[-1]), result["memory_context"]
    assert len(block["static"]) == 1 and len(block["dynamic"]) == 1
    assert block["coverage"]["static"] == {"shown": 1, "omitted": 1, "cut": "count"}
    assert block["coverage"]["dynamic"] == {"shown": 1, "omitted": 0, "cut": None}
    assert context["profile_buckets"]["static"]["cut"] == "count" and context["profile_buckets"]["dynamic"]["cut"] is None
    assert context["profile_truncated"] is True and context["profile_omitted_count"] == 1
    assert context["profile_buckets"]["candidates_truncated"] is False and "reasons" not in block["coverage"]
    await conversation.close()


async def test_standing_buckets_say_what_the_profile_read_left_out(engine, monkeypatch):
    from scone_memory.memory import catalog
    from scone_memory.memory.catalog import BucketBounds

    await dated(engine)
    monkeypatch.setattr(catalog, "MAX_PROFILE_FACTS", 1)
    model = Model()
    conversation = TextConversation(engine, "alpha", "bucket-reasons", lambda: model, standing_profile=ProfilePolicy(),
                                    profile_buckets=BucketBounds())
    result = await conversation.reply("dinner?")
    block = standing_block(model.requests[-1])
    assert "fact_limit" in block["coverage"]["reasons"]
    assert result["memory_context"]["profile_buckets"]["candidates_truncated"] is False
    await conversation.close()


async def test_standing_buckets_with_nothing_to_show_add_nothing(engine):
    from scone_memory.memory.catalog import BucketBounds

    await engine.assert_fact("alpha", "user", "travelling_to", "Kyoto")
    model = Model()
    conversation = TextConversation(engine, "alpha", "bucket-empty", lambda: model, standing_profile=ProfilePolicy(),
                                    profile_buckets=BucketBounds(include="static"))
    result = await conversation.reply("dinner?")
    assert standing_block(model.requests[-1]) is None
    context = result["memory_context"]
    assert context["profile_status"] == "empty" and context["profile_fact_ids"] == []
    assert context["profile_truncated"] is False
    await conversation.close()


async def test_standing_buckets_follow_the_engine_rules(engine):
    from scone_memory.memory.catalog import BucketBounds, BucketRules

    diet, _, trip = await dated(engine)
    engine.profile_bucket_rules = BucketRules.of(dynamic_predicates=["diet"])
    model = Model()
    conversation = TextConversation(engine, "alpha", "bucket-rules", lambda: model, standing_profile=ProfilePolicy(),
                                    profile_buckets=BucketBounds(include="dynamic"))
    result = await conversation.reply("dinner?")
    assert result["memory_context"]["profile_buckets"]["dynamic"]["rules"][str(diet.fact_id)] == "override"
    await conversation.close()


@pytest.mark.parametrize("options, message", [
    ({}, "standing_profile"),
    ({"standing_profile": ProfilePolicy(), "max_profile_bytes": 2000}, "max_profile_bytes"),
    ({"standing_profile": ProfilePolicy(), "profile_limit": 5}, "profile_limit"),
    ({"standing_profile": ProfilePolicy(), "profile_buckets": "both"}, "BucketBounds"),
])
async def test_standing_buckets_refuse_what_they_would_ignore(engine, options, message):
    from scone_memory.memory.catalog import BucketBounds

    options = {"profile_buckets": BucketBounds()} | options
    with pytest.raises(ValueError, match=message):
        TextConversation(engine, "alpha", "bad-buckets", lambda: Model(), **options)
