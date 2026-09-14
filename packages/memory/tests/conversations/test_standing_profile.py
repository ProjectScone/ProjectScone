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
