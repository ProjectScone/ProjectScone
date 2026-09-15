"""Refine and accumulate, under the rule every other synthesis keeps: no sentence without a checked quote.

The reference framework's Refine writes an answer from the first chunk
and rewrites it with each next one; its Accumulate answers every chunk on
its own and joins the answers. Neither names where a sentence came from.
Here both are modes of the evidence-first synthesizer: a refined sentence
must still carry a quote found in a passage read so far, an accumulated
sentence one found in the passage its call was given, and whatever fails
the check is dropped and counted rather than shown.
"""

from __future__ import annotations

import json

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.retrieval import synthesis as module
from scone_memory.retrieval.synthesis import MODES, Passage, SynthesisLimits, synthesize_passages

pytestmark = pytest.mark.asyncio

QUESTION = "What happened with the launch?"
P1 = Passage("chunk:1", "Priya moved the launch to March because the audit ran late.")
P2 = Passage("chunk:2", "Tomas said the audit found two billing errors in February.")
P3 = Passage("chunk:3", "The launch party is booked for the twelfth of March.")
ONE_EACH = SynthesisLimits(max_round_bytes=70)  # every passage above is under 70 bytes and any two are over
# A refine round carries the answer so far inside its byte bound, so its passages are long beside an answer:
# any two padded passages are over 1,500 bytes, and one with a short answer is under.
PAD = " Nothing else was said about it that week." * 24
R1, R2, R3 = (Passage(p.id, p.text + PAD) for p in (P1, P2, P3))
ROUNDS = SynthesisLimits(max_round_bytes=1_500)


def notes(*rows: tuple[str, str, str]) -> str:
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


N1 = ("The launch moved to March.", "chunk:1", "moved the launch to March")
N2 = ("The audit found two billing errors.", "chunk:2", "found two billing errors")
N3 = ("A party is booked in March.", "chunk:3", "booked for the twelfth of March")


async def test_the_modes_are_named():
    assert MODES == ("evidence", "refine", "accumulate")


async def test_an_unknown_mode_is_refused_before_any_call():
    model = FakeChat()
    with pytest.raises(InvalidInput, match="mode"):
        await synthesize_passages(model, QUESTION, [P1], mode="compact")  # type: ignore[arg-type]
    assert model.calls == []


async def test_refine_rewrites_the_answer_with_each_round_and_a_carried_sentence_keeps_its_earlier_quote():
    refined = ("The launch moved to March after an audit.", "chunk:1", "moved the launch to March")
    model = FakeChat([notes(N1), notes(refined, N2)])
    result = await synthesize_passages(model, QUESTION, [R1, R2], limits=ROUNDS, mode="refine")
    assert result.status == "synthesized" and result.folded is False
    assert [s.text for s in result.sentences] == [refined[0], N2[0]], "the refined answer replaces the one before"
    assert [s.citations[0].passage_id for s in result.sentences] == ["chunk:1", "chunk:2"]
    carried = result.sentences[0].citations[0]
    assert R1.text[carried.start:carried.end] == carried.quote, "a quote into an earlier round's passage is checked there"
    assert result.model_calls == 2 and len(model.calls) == 2
    first_system, first_user = model.calls[0]
    assert first_system == module._NOTES_SYSTEM and "[chunk:1] " + P1.text in first_user
    second_system, second_user = model.calls[1]
    assert second_system == module._REFINE_SYSTEM
    assert f'[s1] (chunk:1) {N1[0]} Quote: "{N1[2]}"' in second_user, "the answer so far goes back with its quotes"
    assert "[chunk:2] " + P2.text in second_user and P1.text not in second_user, "earlier passages are not sent again"
    record = result.record()
    assert record["mode"] == "refine" and record["refine_kept_prior"] == 0
    assert record["refine_dropped_carried"] == 0, "a sentence reworded around its own quote was carried, not dropped"
    assert record["reasons"] == []
    assert record["passages"]["cited"] == 2 and record["notes"]["kept"] == 3
    block = f'[s1] (chunk:1) {N1[0]} Quote: "{N1[2]}"'
    assert [r["answer_bytes"] for r in record["rounds"]] == [0, len(block.encode())]


async def test_a_refined_sentence_that_quotes_nothing_is_dropped_and_counted():
    model = FakeChat([notes(N1), notes(N1, ("The launch moved to April.", "chunk:2", "moved the launch to April"))])
    result = await synthesize_passages(model, QUESTION, [R1, R2], limits=ROUNDS, mode="refine")
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.notes_dropped_unquoted == 1


async def test_a_refined_sentence_citing_a_passage_not_yet_read_is_dropped_as_unknown():
    model = FakeChat([notes(N1), notes(N2, ("A party is booked.", "chunk:3", "booked for the twelfth"))])
    limits = SynthesisLimits(max_round_bytes=1_500, max_rounds=2)
    result = await synthesize_passages(model, QUESTION, [R1, R2, R3], limits=limits, mode="refine")
    assert [s.text for s in result.sentences] == [N2[0]]
    assert result.notes_dropped_unknown == 1 and result.passages_unread == 1 and result.status == "partial"


async def test_a_refine_round_whose_sentences_all_fail_leaves_the_answer_standing_and_says_so():
    model = FakeChat([notes(N1), notes(("Wrong.", "chunk:2", "not in the passage"))])
    result = await synthesize_passages(model, QUESTION, [R1, R2], limits=ROUNDS, mode="refine")
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.refine_kept_prior == 1 and result.notes_dropped_unquoted == 1
    assert any("stands" in reason for reason in result.reasons)
    assert result.record()["refine_kept_prior"] == 1


async def test_an_unreadable_refine_round_leaves_the_answer_standing_and_the_rest_still_run():
    model = FakeChat([notes(N1), "I would rather not answer in JSON.", notes(N1, N3)])
    result = await synthesize_passages(model, QUESTION, [R1, R2, R3], limits=ROUNDS, mode="refine")
    assert [r.status for r in result.rounds] == ["noted", "unreadable", "noted"]
    assert result.rounds[1].answer_bytes > 0, "an unreadable refine round still carried the answer"
    assert [s.text for s in result.sentences] == [N1[0], N3[0]]
    assert result.refine_kept_prior == 1 and result.passages_unread == 1 and result.status == "partial"
    assert "round 2: the reply could not be read as notes; the answer so far stands" in result.reasons


async def test_refine_asks_for_an_answer_until_one_exists():
    model = FakeChat([notes(), notes(N2)])
    result = await synthesize_passages(model, QUESTION, [R1, R2], limits=ROUNDS, mode="refine")
    assert [system for system, _ in model.calls] == [module._NOTES_SYSTEM, module._NOTES_SYSTEM]
    assert [s.text for s in result.sentences] == [N2[0]] and result.refine_kept_prior == 0


async def test_a_failing_model_mid_refine_keeps_the_answer_so_far():
    model = FakeChat([notes(N1), ChatError("down")])
    result = await synthesize_passages(model, QUESTION, [R1, R2, R3], limits=ROUNDS, mode="refine")
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.status == "partial" and result.passages_unread == 2
    assert [r.status for r in result.rounds] == ["noted", "failed"] and result.rounds[1].answer_bytes > 0
    assert any(reason.startswith("model failed") for reason in result.reasons)


async def test_refine_never_folds():
    model = FakeChat([notes(N1), notes(N1, N2)])
    result = await synthesize_passages(model, QUESTION, [R1, R2], limits=ROUNDS, mode="refine")
    assert result.model_calls == 2 and result.folded is False and len(result.sentences) == 2


async def test_a_refine_round_holds_the_answer_so_far_inside_its_byte_bound():
    # Without the answer, P1, P2 and P3 (170 bytes) would share the second round of 200 bytes.
    lead = Passage("chunk:0", "z" * 150)
    first = ("Zeds.", "chunk:0", "zzzz")
    short = ("Moved.", "chunk:1", "moved")
    model = FakeChat([notes(first), notes(first, short), notes(first, short, N3)])
    limits = SynthesisLimits(max_round_bytes=200)
    result = await synthesize_passages(model, QUESTION, [lead, P1, P2, P3], limits=limits, mode="refine")
    block = f'[s1] (chunk:0) {first[0]} Quote: "{first[2]}"'
    assert [r.passage_count for r in result.rounds] == [1, 2, 1], "the carried answer pushed P3 to a round of its own"
    assert result.rounds[1].answer_bytes == len(block.encode())
    assert all(r.passage_bytes + r.answer_bytes <= limits.max_round_bytes for r in result.rounds)
    assert P3.text not in model.calls[1][1] and "[chunk:3] " + P3.text in model.calls[2][1]
    assert result.status == "synthesized" and result.truncated is False and len(result.sentences) == 3
    assert [r["answer_bytes"] for r in result.record()["rounds"]][0] == 0


async def test_refine_stops_when_the_answer_so_far_leaves_no_room_and_says_what_it_left_unread():
    long = ("The launch moved to March because the audit ran late, Priya said, " * 2, "chunk:1",
            "moved the launch to March")
    model = FakeChat([notes(long), notes(long, N2)])
    limits = SynthesisLimits(max_round_bytes=100)
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=limits, mode="refine")
    assert result.model_calls == 1 and len(model.calls) == 1, "no call is sent over the bound"
    assert result.passages_read == 1 and result.passages_unread == 2 and result.passages_oversize == 0
    assert result.truncated is True and result.status == "partial"
    block = f'[s1] (chunk:1) {long[0].strip()} Quote: "{long[2]}"'
    assert (f"2 passage(s) unread: the answer so far ({len(block.encode())} bytes) left no room for the next "
            "passage in a round of 100 bytes") in result.reasons


async def test_a_rewrite_that_leaves_out_checked_sentences_counts_them_and_says_so():
    also = ("The audit ran late.", "chunk:1", "the audit ran late")
    model = FakeChat([notes(N1, also), notes(N2)])
    result = await synthesize_passages(model, QUESTION, [R1, R2], limits=ROUNDS, mode="refine")
    assert [s.text for s in result.sentences] == [N2[0]], "the rewrite is the answer"
    assert result.refine_dropped_carried == 2 and result.record()["refine_dropped_carried"] == 2
    assert "round 2: 2 sentence(s) of the answer so far left out of the rewrite" in result.reasons
    kept_one = await synthesize_passages(FakeChat([notes(N1, also), notes(also, N2)]), QUESTION, [R1, R2],
                                         limits=ROUNDS, mode="refine")
    assert kept_one.refine_dropped_carried == 1 and [s.text for s in kept_one.sentences] == [also[0], N2[0]]


async def test_accumulate_answers_one_passage_per_call_and_joins_without_a_fold():
    # Exactly one reply per passage: a fold call would run past the script and raise.
    model = FakeChat([notes(N1), notes(N2), notes(N3)])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], mode="accumulate")
    assert [r.passage_count for r in result.rounds] == [1, 1, 1], "packed one passage a round, under any byte bound"
    assert [s.text for s in result.sentences] == [N1[0], N2[0], N3[0]]
    assert result.model_calls == 3 and result.folded is False and result.status == "synthesized"
    assert all(system == module._NOTES_SYSTEM for system, _ in model.calls)
    assert "[chunk:2] " + P2.text in model.calls[1][1] and P1.text not in model.calls[1][1]
    record = result.record()
    assert record["mode"] == "accumulate" and record["passages"]["cited"] == 3


async def test_an_accumulated_sentence_must_quote_the_passage_its_own_call_was_given():
    model = FakeChat([notes(N1), notes(N2, ("The launch moved.", "chunk:1", "moved the launch to March"))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="accumulate")
    assert [s.text for s in result.sentences] == [N1[0], N2[0]]
    assert result.notes_dropped_unknown == 1


async def test_accumulate_calls_are_bounded_by_rounds_and_the_unread_are_counted():
    model = FakeChat([notes(N1), notes(N2)])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=SynthesisLimits(max_rounds=2),
                                       mode="accumulate")
    assert result.model_calls == 2 and result.passages_read == 2 and result.passages_unread == 1
    assert result.truncated and result.status == "partial"
    assert any("1 passage(s) unread" in reason for reason in result.reasons)


async def test_the_evidence_mode_is_the_default_and_records_its_name():
    model = FakeChat([notes(N1, ("Priya moved it.", "chunk:1", "Priya moved"), N2)])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3])
    record = result.record()
    assert record["mode"] == "evidence" and record["refine_kept_prior"] == 0
    assert len(result.sentences) == 3 and record["passages"]["cited"] == 2, "distinct passages cited, not sentences or passages read"


async def test_an_unreadable_round_outside_refine_has_no_answer_to_keep():
    model = FakeChat(["garbage", notes(N2)])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="accumulate")
    assert result.reasons == ("round 1: the reply could not be read as notes",) and result.refine_kept_prior == 0
