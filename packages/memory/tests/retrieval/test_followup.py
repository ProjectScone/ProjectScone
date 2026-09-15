"""A follow-up turn searched with what the conversation already named.

"Where does Alice Chen work?" and then "since when?": the second turn
names nothing, so a search for it finds nothing about Alice Chen. What
must hold: a turn that leans on the conversation carries the earlier
turn's named or rare terms verbatim into a second query, fused by rank
with the question as asked; a standalone question and a first turn are
left alone; a model's rewrite is taken only when it can be trusted, and
a failed or slow model falls back with its reason; every bound says
when it cut.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import Fact, RecallItem, RecallResult
from scone_memory.core.validation import MAX_QUERY
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.retrieval.followup import (
    MAX_CARRIED_TERMS, MAX_LOOKBACK_TURNS, carry, fused, named_terms, plan_followup, rewrite_followup,
)

ALICE = [{"role": "user", "content": "Where does Alice Chen work?"},
         {"role": "assistant", "content": "At Acme Robotics."}]


def turn(text: str) -> list[dict[str, object]]:
    return [*ALICE, {"role": "user", "content": text}]


def reply(query: str) -> str:
    return json.dumps({"query": query})


class SlowChat:
    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(0.5)
        return reply("late")


def test_named_terms_are_verbatim_runs_quotes_and_rare_shapes_in_order():
    text = ('Where does Alice Chen\'s team keep "Harbor, North" notes on node.js v2 from 2019 '
            'on an iPhone, and does API work?')
    assert named_terms(text) == ["Alice Chen", "Harbor, North", "node.js", "v2", "2019", "iPhone", "API"]


def test_a_name_ends_at_an_opening_mark_at_a_lone_mark_and_at_the_end_of_the_text():
    assert named_terms("we met Alice (Bob Stone) there") == ["Alice", "Bob Stone"]
    assert named_terms("ask Bob ... Alice ?") == ["Bob", "Alice"]
    assert named_terms("ask Alice Chen") == ["Alice Chen"]


def test_an_empty_quotation_is_not_a_term():
    assert named_terms('he said "  " twice') == []


def test_a_sentence_may_open_with_a_name_but_not_with_an_asking_word():
    assert named_terms("Globex hired Carol. Where did she go?") == ["Globex", "Carol"]


def test_punctuation_ends_a_name_and_a_name_is_counted_once():
    assert named_terms("Alice, Bob and ACME met Acme at Bob's.") == ["Alice", "Bob", "ACME"]


def test_since_when_carries_alice_chen_from_the_turn_that_named_her():
    followup = carry(turn("since when?"))
    assert followup.applied is True and followup.method == "carry"
    assert followup.carried == ("Alice Chen",) and followup.from_message == 0
    assert followup.query == "Alice Chen since when?"
    assert "since when" in followup.cues
    record = followup.record()
    assert record["carried"] == ["Alice Chen"] and record["from_message"] == 0 and record["cut"] is False
    assert record["query"] == "Alice Chen since when?" and record["applied"] is True


def test_what_the_assistant_said_is_not_carried():
    assert "Acme Robotics" not in carry(turn("since when?")).carried


def test_a_standalone_second_question_is_left_unchanged_and_says_so():
    followup = carry(turn("Which city is Globex based in?"))
    assert followup.applied is False and followup.query is None and followup.carried == ()
    assert "standalone" in followup.reason


def test_a_first_turn_is_left_unchanged_and_says_so():
    followup = carry([{"role": "user", "content": "since when?"}])
    assert followup.applied is False and followup.query is None and "first turn" in followup.reason


def test_a_referring_word_makes_a_question_that_names_things_lean_on_the_conversation():
    followup = carry(turn("Is that office in Lisbon or Porto?"))
    assert followup.applied is True and followup.carried == ("Alice Chen",) and "that" in followup.cues


def test_a_short_turn_leans_on_the_conversation_even_when_it_names_something():
    followup = carry(turn("Globex too?"))
    assert followup.applied is True and followup.query == "Alice Chen Globex too?" and "short" in followup.cues
    assert carry(turn("Globex hired Carol?")).applied is True
    assert carry(turn("Globex hired Carol Diaz?")).applied is False


def test_a_named_question_with_few_telling_words_still_stands_alone():
    assert carry(turn("Was Bob at Globex?")).applied is False


def test_referring_phrases_lean_on_the_conversation():
    assert "what about" in carry(turn("What about the team tier?")).cues
    assert "how about" in carry(turn("How about the Porto office?")).cues
    assert carry(turn("How about the Porto office?")).applied is True


def test_a_cue_is_named_once():
    assert carry(turn("Is it there or is it gone?")).cues.count("it") == 1


def test_a_turn_naming_nothing_with_one_telling_word_leans_but_two_stand_alone():
    leaning = carry(turn("What did the contract say?"))
    assert leaning.applied is True and "names nothing" in leaning.cues
    standing = carry(turn("What did the contract renewal say?"))
    assert standing.applied is False and "standalone" in standing.reason
    assert carry(turn("What did the menu cost?")).applied is False
    assert "names nothing" in carry(turn("Could you tell me again?")).cues


def test_nothing_is_carried_when_the_question_already_names_it():
    followup = carry(turn("Does Alice Chen still work there?"))
    assert followup.applied is False and followup.query is None and "already" in followup.reason


def test_an_earlier_follow_up_is_skipped_to_reach_the_turn_that_named_something():
    messages = [*turn("since when?"), {"role": "assistant", "content": "2021."}, {"role": "user", "content": "and before that?"}]
    followup = carry(messages)
    assert followup.applied is True and followup.carried == ("Alice Chen",) and followup.from_message == 0
    assert MAX_LOOKBACK_TURNS >= 2


def test_the_most_recent_turn_that_named_something_is_the_one_carried():
    messages = [ALICE[0], {"role": "user", "content": "Where is Globex based?"}, {"role": "user", "content": "since when?"}]
    followup = carry(messages)
    assert followup.carried == ("Globex",) and followup.from_message == 1


def test_blank_or_non_text_user_messages_are_not_turns():
    messages = [*turn("since when?"), {"role": "user", "content": "  "},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/x"}}]}]
    assert carry(messages) == carry(turn("since when?"))
    with pytest.raises(InvalidInput):
        carry([{"role": "assistant", "content": "hello"}])


def test_the_lookback_bound_reports_the_turns_it_did_not_read():
    messages = [*turn("since when?"), {"role": "user", "content": "and before that?"}]
    bounded = carry(messages, lookback=1)
    assert bounded.applied is False and bounded.turns_unread == 1 and "not read" in bounded.reason
    assert bounded.record()["turns_unread"] == 1
    assert carry(messages).turns_unread == 0


def test_the_carried_terms_bound_reports_that_it_cut():
    messages = [{"role": "user", "content": "Did Alice Chen, Bob Stone and Carol Diaz meet?"},
                {"role": "user", "content": "when?"}]
    bounded = carry(messages, max_terms=2)
    assert bounded.carried == ("Alice Chen", "Bob Stone") and bounded.terms_found == 3 and bounded.cut is True
    assert bounded.record()["cut"] is True and bounded.record()["terms_found"] == 3
    whole = carry(messages)
    assert whole.carried == ("Alice Chen", "Bob Stone", "Carol Diaz") and whole.cut is False
    assert MAX_CARRIED_TERMS >= 3


def test_a_carried_query_over_the_query_bound_is_not_searched_and_says_so():
    long = "what did it " + "say " * ((MAX_QUERY - 12) // 4)
    assert len(long) <= MAX_QUERY
    followup = carry(turn(long))
    assert followup.applied is False and followup.query is None and "bound" in followup.reason


def test_the_question_searched_can_be_the_formulated_one():
    followup = carry(turn("since when?"), question="since when")
    assert followup.query == "Alice Chen since when"


async def test_a_readable_rewrite_is_applied_and_the_model_saw_the_history():
    model = FakeChat([reply("Since when has  Alice Chen worked at Acme Robotics?")])
    messages = [{"role": "system", "content": "Be terse."}, *ALICE, {"role": "assistant", "content": "  "},
                {"role": "user", "content": "since when?"}]
    followup = await rewrite_followup(model, messages)
    assert followup.applied is True and followup.method == "rewrite" and followup.model_calls == 1
    assert followup.query == "Since when has Alice Chen worked at Acme Robotics?" and followup.fallback is None
    system, user = model.calls[0]
    assert "Where does Alice Chen work?" in user and "At Acme Robotics." in user and "since when?" in user
    assert "Be terse." not in user and user.count("ASSISTANT:") == 1
    assert "JSON" in system


async def test_a_rewrite_equal_to_the_question_leaves_it_as_asked():
    followup = await rewrite_followup(FakeChat([reply("Since  when?")]), turn("since when?"))
    assert followup.applied is False and followup.method == "rewrite" and followup.query is None
    assert "standalone" in followup.reason and followup.fallback is None


async def test_a_failed_model_falls_back_to_carrying_with_the_reason():
    followup = await rewrite_followup(FakeChat([ChatError("down")]), turn("since when?"))
    assert followup.method == "carry" and followup.applied is True and followup.carried == ("Alice Chen",)
    assert followup.fallback is not None and followup.fallback.startswith("model failed: ChatError")
    assert followup.model_calls == 1 and followup.record()["fallback"] == followup.fallback


async def test_a_slow_model_falls_back_with_a_timeout_reason():
    followup = await rewrite_followup(SlowChat(), turn("since when?"), timeout_s=0.05)
    assert followup.method == "carry" and followup.applied is True
    assert followup.fallback is not None and "timeout" in followup.fallback


async def test_a_failed_model_on_a_standalone_question_falls_back_to_the_plain_question():
    followup = await rewrite_followup(FakeChat([ChatError("down")]), turn("Which city is Globex based in?"))
    assert followup.method == "none" and followup.applied is False and followup.query is None
    assert followup.fallback is not None and "standalone" in followup.reason


@pytest.mark.parametrize("answer, reason", [
    ("I would rather not.", "could not be read"),
    (reply(""), "could not be read"),
    (reply("weather forecast tomorrow"), "shares no word"),
    (reply("Alice " + "x" * MAX_QUERY), "bound"),
])
async def test_a_rewrite_that_cannot_be_trusted_falls_back_with_the_reason(answer, reason):
    followup = await rewrite_followup(FakeChat([answer]), turn("since when?"))
    assert followup.method == "carry" and followup.fallback is not None and reason in followup.fallback


async def test_a_rewrite_may_share_a_word_with_the_conversation_or_with_the_question():
    followup = await rewrite_followup(FakeChat([reply("Alice Chen employment start")]), turn("since when?"))
    assert followup.applied is True and followup.method == "rewrite"
    followup = await rewrite_followup(FakeChat([reply("employment start since")]), turn("since when?"))
    assert followup.applied is True and followup.method == "rewrite"


async def test_a_first_turn_asks_no_model():
    model = FakeChat()
    followup = await rewrite_followup(model, [{"role": "user", "content": "since when?"}])
    assert model.calls == [] and followup.applied is False and "first turn" in followup.reason
    assert followup.method == "none" and followup.fallback is None and followup.mode == "rewrite"


async def test_the_history_bound_reports_the_turns_it_left_out():
    messages = [{"role": "user", "content": "Globex?"}, {"role": "user", "content": "Tell me about Initech. " * 20},
                *turn("since when?")]
    model = FakeChat([reply("Since when has Alice Chen worked there?")])
    bounded = await rewrite_followup(model, messages, max_history_bytes=120)
    assert bounded.applied is True and bounded.history_omitted == 2 and bounded.record()["history_omitted"] == 2
    assert "Initech" not in model.calls[0][1] and "Globex" not in model.calls[0][1]
    unbounded = await rewrite_followup(FakeChat([reply("Since when has Alice Chen worked there?")]), messages)
    assert unbounded.history_omitted == 0


async def test_history_that_cannot_fit_at_all_asks_no_model_and_falls_back():
    model = FakeChat()
    followup = await rewrite_followup(model, turn("since when?"), max_history_bytes=10)
    assert model.calls == [] and followup.method == "carry" and followup.fallback is not None
    assert "history" in followup.fallback and followup.model_calls == 0


async def test_plan_followup_gives_the_model_its_deadline():
    followup = await plan_followup(turn("since when?"), "rewrite", model=SlowChat(), timeout_s=0.05)
    assert followup.fallback is not None and "timeout" in followup.fallback


async def test_plan_followup_checks_its_mode_and_model():
    with pytest.raises(InvalidInput):
        await plan_followup(turn("since when?"), "sometimes")
    with pytest.raises(InvalidInput):
        await plan_followup(turn("since when?"), "sometimes", model=FakeChat())
    with pytest.raises(InvalidInput):
        await plan_followup(turn("since when?"), "rewrite")
    assert (await plan_followup(turn("since when?"), "carry")).carried == ("Alice Chen",)
    rewritten = await plan_followup(turn("since when?"), "rewrite", model=FakeChat([reply("Alice Chen since")]))
    assert rewritten.method == "rewrite" and rewritten.query == "Alice Chen since"


def item(chunk_id: int) -> RecallItem:
    return RecallItem(chunk_id=chunk_id, episode_id=chunk_id, text=f"passage {chunk_id}", score=1.0,
                      created_at="2026-01-01T00:00:00Z")


def fact(fact_id: int) -> Fact:
    return Fact(fact_id=fact_id, space="s", subject="alice chen", predicate="works_at", object=f"org {fact_id}",
                valid_from="2026-01-01T00:00:00Z")


def test_two_recalls_are_interleaved_by_rank_the_question_first_each_passage_once():
    original = RecallResult(event_id=7, items=[item(1), item(2), item(3)], facts=[fact(10), fact(12)], low_confidence=True,
                            degraded=["vectors: down"], top_similarity=0.2)
    second = RecallResult(event_id=8, items=[item(4), item(3), item(2), item(5)], facts=[fact(11), fact(10)],
                          low_confidence=False, degraded=["text: slow"], top_similarity=0.6)
    both = fused(original, second, limit=4)
    # Reciprocal rank fusion would put 2 and 3 first: both lists hold them.
    assert [i.chunk_id for i in both.items] == [1, 4, 2, 3]
    assert [i.chunk_id for i in fused(original, second, limit=5).items] == [1, 4, 2, 3, 5]
    assert [f.fact_id for f in both.facts] == [10, 11, 12]
    assert both.event_id == 7 and both.low_confidence is False and both.top_similarity == 0.6
    assert both.degraded == ["text: slow", "vectors: down"]


def test_fused_confidence_is_low_only_when_both_are_low_and_unknown_otherwise():
    low = RecallResult(low_confidence=True)
    unknown = RecallResult(low_confidence=None)
    assert fused(low, low, limit=5).low_confidence is True
    assert fused(low, unknown, limit=5).low_confidence is None
    assert fused(unknown, unknown, limit=5).top_similarity is None
