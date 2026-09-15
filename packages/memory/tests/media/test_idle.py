"""The idle watch: a user who has gone quiet, with the time passed in as plain numbers."""

from __future__ import annotations

import math

import pytest

from scone_memory.realtime.idle import END_AFTER, PROMPT, IdlePolicy, IdleReceipt, IdleWatch


@pytest.mark.parametrize("options", [
    {"timeout": 0}, {"timeout": -1}, {"timeout": True}, {"timeout": math.inf}, {"timeout": math.nan},
    {"timeout": "5"}, {"timeout": 5, "prompt": ""}, {"timeout": 5, "prompt": "   "}, {"timeout": 5, "prompt": 7},
    {"timeout": 5, "end_after": 0}, {"timeout": 5, "end_after": True}, {"timeout": 5, "end_after": 1.5},
    {"timeout": 5, "prompt": None, "end_after": None},
])
def test_a_policy_that_cannot_work_is_refused(options):
    with pytest.raises(ValueError):
        IdlePolicy(**options)


def test_the_defaults_prompt_and_end_after_three():
    policy = IdlePolicy(8)
    assert (policy.prompt, policy.end_after) == (PROMPT, END_AFTER) == ("Are you still there?", 3)


def test_the_wait_starts_with_the_conversation_and_an_idle_is_the_timeout_later():
    watch = IdleWatch(IdlePolicy(5), 100.0)
    assert watch.deadline == 105.0
    assert watch.expire(104.9) is None
    assert watch.expire(105.0) == IdleReceipt(1, "prompt", 5000.0)
    assert watch.deadline == 110.0, "with nothing said, the next idle is a timeout after this one"
    assert watch.expire(110.0) == IdleReceipt(2, "prompt", 5000.0)


def test_speech_stops_the_wait_and_its_end_starts_it_again_with_the_count_cleared():
    watch = IdleWatch(IdlePolicy(5, end_after=None), 100.0)
    assert watch.expire(105.0).count == 1
    watch.speaking(107.0)
    assert watch.deadline is None and watch.count == 0
    assert watch.expire(200.0) is None, "a user still speaking is not idle, however long they speak"
    watch.spoke(208.0)
    assert watch.deadline == 213.0
    assert watch.expire(213.0) == IdleReceipt(1, "prompt", 5000.0)


def test_words_or_keys_without_a_speech_start_restart_the_wait():
    watch = IdleWatch(IdlePolicy(5, end_after=None), 100.0)
    watch.expire(105.0)
    watch.spoke(107.0)
    assert (watch.deadline, watch.count) == (112.0, 0)


def test_a_user_mid_turn_starts_the_wait_again_without_clearing_the_count_or_their_speech():
    watch = IdleWatch(IdlePolicy(5, end_after=None), 100.0)
    watch.expire(105.0)
    watch.restart(107.0)
    assert (watch.deadline, watch.count) == (112.0, 1)
    watch.speaking(108.0)
    watch.restart(109.0)
    assert watch.deadline is None, "still speaking"
    watch.spoke(110.0)
    watch.pause()
    watch.restart(111.0)
    assert watch.deadline is None, "still paused"


def test_the_bot_speaking_pauses_the_wait_and_it_restarts_when_the_bot_stops():
    watch = IdleWatch(IdlePolicy(5), 100.0)
    watch.pause()
    assert watch.deadline is None and watch.expire(1000.0) is None
    watch.resume(120.0)
    assert watch.deadline == 125.0
    assert watch.expire(125.0) == IdleReceipt(1, "prompt", 5000.0), "silence is counted from when the bot stopped"


def test_pauses_are_counted_so_a_tool_and_the_bot_both_hold_it():
    watch = IdleWatch(IdlePolicy(5), 100.0)
    watch.pause()
    watch.pause()
    watch.resume(110.0)
    assert watch.deadline is None, "one of two pauses is still on"
    watch.resume(111.0)
    assert watch.deadline == 116.0
    with pytest.raises(RuntimeError):
        watch.resume(112.0)


def test_speech_ending_during_a_pause_does_not_start_the_wait():
    watch = IdleWatch(IdlePolicy(5), 100.0)
    watch.pause()
    watch.speaking(101.0)
    watch.spoke(102.0)
    assert watch.deadline is None
    watch.resume(103.0)
    assert watch.deadline == 108.0


def test_the_bot_stopping_while_the_user_speaks_does_not_start_the_wait():
    watch = IdleWatch(IdlePolicy(5), 100.0)
    watch.pause()
    watch.speaking(101.0)
    watch.resume(102.0)
    assert watch.deadline is None
    watch.spoke(104.0)
    assert watch.deadline == 109.0


def test_the_last_idle_before_the_bound_ends_and_nothing_follows_it():
    watch = IdleWatch(IdlePolicy(5, end_after=2), 100.0)
    assert watch.expire(105.0).action == "prompt"
    assert watch.expire(110.0) == IdleReceipt(2, "end", 5000.0)
    assert watch.ended and watch.deadline is None
    watch.spoke(111.0)
    watch.pause()
    watch.resume(112.0)
    assert watch.deadline is None and watch.expire(1e9) is None


def test_end_after_one_ends_at_the_first_idle():
    assert IdleWatch(IdlePolicy(5, prompt=None, end_after=1), 0.0).expire(5.0) == IdleReceipt(1, "end", 5000.0)


def test_without_an_end_the_count_keeps_climbing():
    watch = IdleWatch(IdlePolicy(1, end_after=None), 0.0)
    receipts = [watch.expire(float(second)) for second in range(1, 11)]
    assert [r.count for r in receipts] == list(range(1, 11))
    assert {r.action for r in receipts} == {"prompt"} and not watch.ended


def test_without_a_prompt_an_idle_is_noted_and_nothing_is_said():
    watch = IdleWatch(IdlePolicy(5, prompt=None, end_after=3), 0.0)
    assert [watch.expire(t).action for t in (5.0, 10.0, 15.0)] == ["noted", "noted", "end"]


def test_silence_is_kept_to_the_thousandth_of_a_millisecond():
    assert IdleWatch(IdlePolicy(5), 0.1).expire(5.2).silent_ms == 5100.0


def test_a_receipt_is_record_metadata():
    assert IdleReceipt(2, "prompt", 5012.4).metadata() == \
        {"idle_count": "2", "idle_action": "prompt", "idle_silent_ms": "5012"}
