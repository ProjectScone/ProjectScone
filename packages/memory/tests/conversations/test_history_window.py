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

import asyncio
import re

import pytest

from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.realtime.text import TextConversation, _bytes


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


class Summariser:
    """Folds the turns it is shown into one line naming them; scripted
    otherwise, and slow by ``delay`` seconds when asked."""

    def __init__(self, answers=None, fail=False, delay=0.0):
        self.answers, self.fail, self.delay, self.calls = list(answers or []), fail, delay, []

    async def complete(self, system, user):
        self.calls.append((system, user))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("summariser down")
        if self.answers:
            return self.answers.pop(0)
        turns = sorted(set(re.findall(r"User: (turn \d+)", user)))
        earlier = re.search(r"Summary so far:\n(.*?)\n\nTurns leaving", user, flags=re.S).group(1)
        kept = earlier if earlier != "(nothing yet)" else ""
        return (kept + " " if kept else "") + "Covered " + ", ".join(turns) + "."


async def test_a_summary_folds_the_turns_that_left_into_the_system_message(engine):
    summariser = Summariser()
    conversation, model = talk(engine, "summary", 10, max_history_bytes=2000, history_policy="summary",
                               summary_factory=lambda: summariser, max_summary_bytes=400)
    receipts = [await conversation.reply(f"turn {n} " + FILLER) for n in range(10)]
    first, last = receipts[0]["history"], receipts[-1]["history"]
    assert first["summary"] == {"status": "none", "bytes": 0, "max_bytes": 400, "timeout_s": 20.0, "turn_count": 0,
                                "turn_ids": []}
    assert last["policy"] == "summary" and last["evicted_turn_count"] >= 1 and last["bytes"] <= 2000
    left = [turn for receipt in receipts for turn in receipt["history"]["evicted_turn_ids"]]
    assert left and last["summary"]["status"] == "summarised" and last["summary"]["turn_ids"] == left, \
        "every turn that left, over the whole conversation, was folded into the summary, in order"
    assert last["summary"]["turn_count"] == len(left)
    assert 0 < last["summary"]["bytes"] <= 400
    seen = model.requests[-1]
    assert seen[0]["role"] == "system" and seen[0]["content"].startswith("Answer.\n\nEarlier in this conversation, summarised:\n")
    assert "Covered turn 0" in seen[0]["content"] and not any(m["content"].startswith("turn 0 ") for m in seen[1:]), \
        "the oldest turn is in the summary and out of the window"
    assert seen[-1]["content"].startswith("turn 9 ")
    assert summariser.calls and "Summary so far:" in summariser.calls[-1][1] and "summary alone" in summariser.calls[-1][0]
    assert summariser.calls[-1][1].count("User: turn") >= 1, "the summariser is shown the turns leaving, with their replies"
    await conversation.close()
    rows = await engine.episodes("alpha", {"session_id": "summary"})
    assert len(rows) == 20, "every turn was captured; the summary is context, not a record"


async def test_a_summary_that_does_not_fit_or_fails_keeps_the_one_before_and_says_so(engine):
    # One good answer, then answers too long for the reserve and empty ones
    # for every fold after it.
    summariser = Summariser(answers=["Covered the first turns.", *["z" * 5000, ""] * 12])
    conversation, model = talk(engine, "summary-bounds", 12, max_history_bytes=2000, history_policy="summary",
                               summary_factory=lambda: summariser, max_summary_bytes=300)
    receipts = [(await conversation.reply(f"turn {n} " + FILLER))["history"]["summary"] for n in range(12)]
    statuses = [receipt["status"] for receipt in receipts]
    assert "summarised" in statuses and "too_long" in statuses and "empty" in statuses, statuses
    assert model.requests[-1][0]["content"].endswith("Covered the first turns."), \
        "an answer too long for its reserve, or empty, keeps the summary before it"
    assert receipts[-1]["turn_count"] == receipts[statuses.index("summarised")]["turn_count"], \
        "turns whose fold was refused are not counted as summarised"
    assert all(_bytes(request) <= 2000 for request in model.requests), "the summary never pushes the history past its limit"
    await conversation.close()
    broken = Summariser(fail=True)
    conversation, _ = talk(engine, "summary-failed", 6, max_history_bytes=2000, history_policy="summary",
                           summary_factory=lambda: broken, max_summary_bytes=300)
    receipts = [await conversation.reply(f"turn {n} " + FILLER) for n in range(6)]
    assert receipts[-1]["history"]["summary"]["status"] == "failed: RuntimeError" and receipts[-1]["text"].startswith("reply 5")
    assert receipts[-1]["history"]["evicted_turn_count"] >= 1, "the turns still left, as under the window; the summariser failing does not fail the turn"
    await conversation.close()


def test_the_summary_policy_needs_a_model_and_room_for_its_summary(engine):
    with pytest.raises(ValueError, match="summary_factory"):
        TextConversation(engine, "alpha", "no-model", unused_factory, history_policy="summary")
    with pytest.raises(ValueError, match="at least half"):
        TextConversation(engine, "alpha", "no-room", unused_factory, history_policy="summary",
                         summary_factory=lambda: Summariser(), max_history_bytes=2000, max_summary_bytes=1500)
    with pytest.raises(ValueError, match="max_summary_bytes"):
        TextConversation(engine, "alpha", "tiny", unused_factory, history_policy="summary",
                         summary_factory=lambda: Summariser(), max_summary_bytes=10)


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


async def test_the_summary_is_counted_as_the_history_counts_bytes_and_the_prompt_must_leave_it_room(engine):
    """The reserve is bytes as the history counts them (the JSON the model
    is sent), not the summary's characters: a summary of quotation marks
    costs twice its length there. And the system prompt has to fit what
    the reserve leaves, or no turn could ever fit."""
    with pytest.raises(ValueError, match="system prompt exceeds history byte limit"):
        TextConversation(engine, "alpha", "prompt-too-big", unused_factory, system_prompt="P" * 1400,
                         max_history_bytes=2000, max_summary_bytes=800, history_policy="summary",
                         summary_factory=lambda: Summariser(), tool_model_factory=lambda: Script())
    TextConversation(engine, "alpha", "prompt-fits", unused_factory, system_prompt="P" * 1400,
                     max_history_bytes=2000, max_summary_bytes=800, history_policy="window",
                     tool_model_factory=lambda: Script())
    quoted, plain = '"' * 250, "a" * 250
    summariser = Summariser(answers=[quoted, plain, *[plain] * 12])
    conversation, model = talk(engine, "summary-json", 12, max_history_bytes=2000, history_policy="summary",
                               summary_factory=lambda: summariser, max_summary_bytes=300)
    receipts = [(await conversation.reply(f"turn {n} " + FILLER))["history"]["summary"] for n in range(12)]
    statuses = [receipt["status"] for receipt in receipts]
    assert "too_long" in statuses and "summarised" in statuses, statuses
    first_fold = statuses.index("too_long")
    assert statuses[first_fold + 1] == "summarised", "250 quotation marks cost 500 bytes; 250 letters fit"
    kept = receipts[statuses.index("summarised")]
    carrying = next(request[0] for request in model.requests if request[0]["content"].startswith("Answer.\n\n"))
    assert kept["bytes"] == _bytes([carrying]) - _bytes([{"role": "system", "content": "Answer."}]) \
        and 250 < kept["bytes"] <= 300, "the receipt's bytes are the summary's cost in the request"
    assert all(_bytes(request) <= 2000 for request in model.requests), "turns in the room, summary in its reserve"
    await conversation.close()


async def test_a_slow_summariser_keeps_the_summary_before_and_the_turn_still_answers(engine):
    slow = Summariser(delay=0.5)
    conversation, _ = talk(engine, "summary-slow", 6, max_history_bytes=2000, history_policy="summary",
                           summary_factory=lambda: slow, max_summary_bytes=300, summary_timeout=0.1)
    receipts = [await conversation.reply(f"turn {n} " + FILLER) for n in range(6)]
    assert receipts[-1]["history"]["summary"]["status"] == "timed_out" and receipts[-1]["text"].startswith("reply 5")
    assert receipts[-1]["history"]["summary"]["timeout_s"] == 0.1 and receipts[-1]["history"]["evicted_turn_count"] >= 1
    await conversation.close()
    with pytest.raises(ValueError, match="summary_timeout"):
        talk(engine, "summary-bad-timeout", 1, history_policy="summary", summary_factory=lambda: slow,
             summary_timeout=0)


async def test_a_reply_that_overflows_is_folded_too_and_a_failed_turn_folds_nothing(engine):
    """The post-reply eviction folds as the pre-reply one does; and a turn
    that fails after folding leaves the summary as it was, so the turns
    it folded are folded once, by the turn that commits."""
    summariser = Summariser()
    model = Script(ToolStep(content="short"), ToolStep(content="z" * 1200))
    conversation = TextConversation(engine, "alpha", "summary-reply-overflow", unused_factory, system_prompt="Answer.",
                                    max_history_bytes=2000, history_policy="summary", max_summary_bytes=300,
                                    summary_factory=lambda: summariser, tool_model_factory=lambda: model)
    await conversation.reply("turn 0 " + FILLER)
    second = await conversation.reply("turn 1 " + "w" * 100)
    assert second["text"].startswith("z") and second["history"]["evicted_turn_count"] == 1
    assert second["history"]["summary"]["status"] == "summarised" and "turn 0" in summariser.calls[-1][1], \
        "the turn the reply pushed out was folded after the reply"
    await conversation.close()
    failing = Summariser()

    async def broken_reply():
        raise RuntimeError("model down")

    model = Script(*[ToolStep(content=f"reply {n} " + "y" * 200) for n in range(4)], broken_reply,
                   ToolStep(content="reply 5 after"))
    conversation = TextConversation(engine, "alpha", "summary-failed-turn", unused_factory, system_prompt="Answer.",
                                    max_history_bytes=2000, history_policy="summary", max_summary_bytes=300,
                                    summary_factory=lambda: failing, tool_model_factory=lambda: model)
    receipts = [await conversation.reply(f"turn {n} " + FILLER) for n in range(4)]
    assert receipts[-1]["history"]["summary"]["status"] == "summarised"
    folds_before = len(failing.calls)
    committed = receipts[-1]["history"]["summary"]["turn_ids"]
    left = {turn for receipt in receipts for turn in receipt["history"]["evicted_turn_ids"]}
    with pytest.raises(RuntimeError, match="tool model unavailable"):
        # A message this big pushes every earlier turn out, so this turn folds.
        await conversation.reply("turn 4 " + "x" * 1200)
    assert len(failing.calls) == folds_before + 1, "the failed turn asked for its fold before the model failed"
    assert conversation._summary.turns == committed and conversation._evicted_turns == left, \
        "the failed turn's fold and evictions did not commit; the summary and the window's record are as they were"
    await conversation.close()
