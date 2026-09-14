"""A long conversation continues on a window instead of ending in a refusal.

Today a conversation whose history reaches the byte limit raises "start a
new conversation", and a reply that would push it over fails the turn.
Every turn is already captured as its own episode, so nothing is lost by
letting the oldest turns leave the model's context: under the `window`
policy the conversation evicts whole oldest turns until the next one
fits, never leaving the window starting on an assistant message, keeps
the newest turn whole, and says in each receipt what it evicted. A single
message that cannot fit on its own is still refused, explicitly. The
default policy is unchanged.
"""
from __future__ import annotations

import pytest

from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.realtime.text import TextConversation


class Script:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    async def complete(self, messages, tools):
        self.requests.append(messages)
        step = self.steps.pop(0)
        return await step() if callable(step) else step


def unused_factory():
    raise AssertionError("tool mode must not create the ordinary text provider")


FILLER = "x" * 300


def talk(engine, session_id, turns, **options):
    model = Script(*[ToolStep(content=f"reply {n} " + "y" * 200) for n in range(turns)])
    return TextConversation(engine, "alpha", session_id, unused_factory, system_prompt="Answer.",
                            tool_model_factory=lambda: model, **options), model


async def test_the_default_still_refuses_at_the_limit(engine):
    conversation, _ = talk(engine, "refuse", 10, max_history_bytes=2000)
    with pytest.raises(ValueError, match="conversation history byte limit"):
        for n in range(10):
            await conversation.reply(f"turn {n} " + FILLER)
    await conversation.close()


async def test_a_window_evicts_whole_oldest_turns_and_says_so(engine):
    conversation, model = talk(engine, "window", 10, max_history_bytes=2000, history_policy="window")
    receipts = [await conversation.reply(f"turn {n} " + FILLER) for n in range(10)]
    first, last = receipts[0]["history"], receipts[-1]["history"]
    assert first == {"policy": "window", "max_bytes": 2000, "bytes": first["bytes"], "evicted_turn_count": 0, "evicted_turn_ids": []}
    assert last["policy"] == "window" and last["evicted_turn_count"] >= 1 and last["bytes"] <= 2000
    assert len(last["evicted_turn_ids"]) == last["evicted_turn_count"]
    # What the model saw on the last turn: the system prompt, then whole turns, the newest last.
    seen = model.requests[-1]
    assert [m["role"] for m in seen[:2]] == ["system", "user"]
    assert seen[-1]["content"].startswith("turn 9 ")
    assert not any(m["content"].startswith("turn 0 ") for m in seen), "the oldest turn left the window"
    await conversation.close()
    rows = await engine.episodes("alpha", {"session_id": "window"})
    assert len(rows) == 20, "every turn was captured, evicted or not"


async def test_a_single_message_over_the_window_is_still_refused(engine):
    conversation, _ = talk(engine, "big", 1, max_history_bytes=2000, history_policy="window")
    with pytest.raises(ValueError, match="one message"):
        await conversation.reply("m" * 2500)
    await conversation.close()
    assert await engine.episodes("alpha", {"session_id": "big"}) == []


async def test_a_reply_that_overflows_evicts_rather_than_failing_the_turn(engine):
    model = Script(ToolStep(content="short"), ToolStep(content="z" * 1500))
    conversation = TextConversation(engine, "alpha", "reply-overflow", unused_factory, system_prompt="Answer.",
                                    max_history_bytes=2000, history_policy="window", tool_model_factory=lambda: model)
    await conversation.reply("first " + FILLER)
    second = await conversation.reply("second")
    assert second["text"].startswith("z") and second["history"]["evicted_turn_count"] == 1
    await conversation.close()


async def test_a_reply_too_big_for_the_window_on_its_own_still_fails(engine):
    model = Script(ToolStep(content="z" * 2500))
    conversation = TextConversation(engine, "alpha", "reply-too-big", unused_factory, system_prompt="Answer.",
                                    max_history_bytes=2000, history_policy="window", tool_model_factory=lambda: model)
    with pytest.raises(RuntimeError, match="model reply exceeded"):
        await conversation.reply("hello")
    await conversation.close()


def test_an_unknown_policy_is_refused(engine):
    with pytest.raises(ValueError, match="history_policy"):
        TextConversation(engine, "alpha", "bad", unused_factory, history_policy="drop")


class Model:
    """A plain text model: records each request, replies with filler."""

    def __init__(self):
        self.requests = []

    async def respond(self, request):
        from scone_memory.realtime.events import ReplyCompleted, TextDelta

        self.requests.append(request)
        yield TextDelta("noted " + "y" * 200)
        yield ReplyCompleted()

    async def aclose(self):
        pass


async def test_an_evicted_turn_comes_back_through_same_session_recall(engine):
    """Same-session turns are kept out of recall because the model already
    holds them. Once a turn has left the window the model no longer does,
    and the memory context admits exactly those turns, saying how many."""
    model = Model()
    conversation = TextConversation(engine, "alpha", "admit", lambda: model, system_prompt="Answer.",
                                    max_history_bytes=2000, history_policy="window")
    first = await conversation.reply("The gate code for the yard is 4471, remember it.")
    assert first["memory_context"]["same_session_items"] == 0
    receipts = [await conversation.reply(f"filler turn {n} " + FILLER) for n in range(6)]
    assert first["turn_id"] in {t for r in receipts for t in r["history"]["evicted_turn_ids"]}, "the first turn left the window"
    asked = await conversation.reply("What is the gate code for the yard?")
    context = asked["memory_context"]
    assert context["same_session_items"] >= 1, context
    assert any(ref["episode_id"] == first["user_episode_id"] for ref in context["references"]), context["references"]
    seen = model.requests[-1]
    [block] = [m["content"] for m in seen if m["content"].startswith("Scone retrieved source material")]
    assert "4471" in block, block[:300]
    assert not any(m["content"].startswith("The gate code") for m in seen), "the turn itself is out of the window"
    await conversation.close()


async def test_a_turn_still_in_the_window_is_not_recalled_twice(engine):
    model = Model()
    conversation = TextConversation(engine, "alpha", "twice", lambda: model, system_prompt="Answer.",
                                    max_history_bytes=20000, history_policy="window")
    first = await conversation.reply("The gate code for the yard is 4471, remember it.")
    asked = await conversation.reply("What is the gate code for the yard?")
    context = asked["memory_context"]
    assert context["same_session_items"] == 0
    assert not any(ref["episode_id"] == first["user_episode_id"] for ref in context["references"])
    await conversation.close()


async def test_on_the_streaming_path_a_reply_too_big_for_the_window_fails_and_one_that_overflows_evicts(engine):
    """The tool path and the streaming path check the reply in different
    places; both must hold. Found by the full suite after a proof run only
    on the tool path let the streaming check be deleted as dead."""
    class Sized:
        def __init__(self, *sizes):
            self.sizes = list(sizes)

        async def respond(self, request):
            from scone_memory.realtime.events import ReplyCompleted, TextDelta

            yield TextDelta("z" * self.sizes.pop(0))
            yield ReplyCompleted()

        async def aclose(self):
            pass

    too_big = TextConversation(engine, "alpha", "stream-too-big", lambda: Sized(2500), system_prompt="Answer.",
                               max_history_bytes=2000, history_policy="window")
    with pytest.raises(RuntimeError, match="history byte limit"):
        await too_big.reply("hello")
    await too_big.close()
    model = Sized(10, 1500)
    overflow = TextConversation(engine, "alpha", "stream-overflow", lambda: model, system_prompt="Answer.",
                                max_history_bytes=2000, history_policy="window")
    await overflow.reply("first " + FILLER)
    second = await overflow.reply("second")
    assert second["history"]["evicted_turn_count"] == 1
    await overflow.close()
