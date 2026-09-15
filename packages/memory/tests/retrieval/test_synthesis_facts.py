"""Facts first, then an answer that may say only what the quoted facts say.

The ``facts`` mode reads each passage on its own and asks the model for
the atomic facts it states that bear on the question, each with a quote
from that passage; a fact whose quote is not in its passage is dropped
and counted. A second call writes the answer from the checked facts
alone, every sentence naming the facts it uses, and a sentence that names
none is dropped and counted. The record says how many calls were made,
how many facts were extracted, how many the answer used, and how many
sentences it lost.
"""

from __future__ import annotations

import json

import pytest

from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.retrieval import synthesis as module
from scone_memory.retrieval.synthesis import Passage, SynthesisLimits, synthesize_passages

pytestmark = pytest.mark.asyncio

QUESTION = "What happened with the launch?"
P1 = Passage("chunk:1", "Priya moved the launch to March because the audit ran late.")
P2 = Passage("chunk:2", "Tomas said the audit found two billing errors in February.")
P3 = Passage("chunk:3", "The launch party is booked for the twelfth of March.")

F1 = ("The launch moved to March.", "moved the launch to March")
F2 = ("The audit found two billing errors.", "found two billing errors")
F3 = ("A party is booked in March.", "booked for the twelfth of March")


def facts(*rows: tuple[str, str]) -> str:
    return json.dumps({"facts": [{"fact": fact, "quote": quote} for fact, quote in rows]})


def answer(*rows: tuple[str, list[str]]) -> str:
    return json.dumps({"answer": [{"sentence": sentence, "facts": ids} for sentence, ids in rows]})


def line(number: int, fact: tuple[str, str]) -> str:
    return f'[f{number}] {fact[0]} Quote: "{fact[1]}"'


async def test_each_passage_is_one_extraction_call_and_the_answer_is_written_from_the_quoted_facts():
    written = "The launch moved to March after an audit found two billing errors."
    model = FakeChat([facts(F1), facts(F2), answer((written, ["f1", "f2"]))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="facts")
    assert result.status == "synthesized" and result.folded is True
    assert [s.text for s in result.sentences] == [written]
    cited = result.sentences[0].citations
    assert [(c.passage_id, c.quote) for c in cited] == [("chunk:1", F1[1]), ("chunk:2", F2[1])]
    assert all(p.text[c.start:c.end] == c.quote for p, c in zip((P1, P2), cited)), "each quote is found where it says"
    assert [r.passage_count for r in result.rounds] == [1, 1], "one passage a call, under any byte bound"
    assert [system for system, _ in model.calls] == [module._FACTS_SYSTEM, module._FACTS_SYSTEM, module._ANSWER_SYSTEM]
    first, second, written_from = (user for _, user in model.calls)
    assert QUESTION in first and P1.text in first and P2.text not in first
    assert P2.text in second and P1.text not in second
    assert QUESTION in written_from and line(1, F1) in written_from and line(2, F2) in written_from
    assert P1.text not in written_from and P2.text not in written_from, "the answer sees the facts, not the passages"
    assert written_from.endswith("\n\nAnswer in at most 24 sentence(s)."), "the sentence bound is asked for, not only cut"
    assert result.model_calls == 3 and result.notes_kept == 2
    record = result.record()
    assert record["mode"] == "facts" and record["model_calls"] == 3
    assert record["facts"] == {"extracted": 2, "used": 2, "unsent": 0, "sentences_dropped_uncited": 0}
    assert record["passages"]["cited"] == 2 and record["reasons"] == []


async def test_a_fact_whose_quote_is_not_in_its_passage_is_dropped_counted_and_never_reaches_the_answer():
    wrong = ("The launch moved to April.", "moved the launch to April")
    model = FakeChat([facts(F1, wrong), answer(("It moved to March.", ["f1"]))])
    result = await synthesize_passages(model, QUESTION, [P1], mode="facts")
    assert result.notes_dropped_unquoted == 1 and result.notes_kept == 1
    assert wrong[0] not in model.calls[1][1] and "[f2]" not in model.calls[1][1]
    assert result.record()["facts"]["extracted"] == 1


async def test_a_fact_may_not_quote_a_passage_its_call_did_not_hold():
    borrowed = ("The audit found errors.", "found two billing errors")
    model = FakeChat([facts(F1, borrowed), facts(F2), answer(("Moved and audited.", ["f1", "f2"]))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="facts")
    assert result.notes_dropped_unquoted == 1 and result.notes_kept == 2
    assert [c.passage_id for c in result.sentences[0].citations] == ["chunk:1", "chunk:2"]


async def test_malformed_facts_are_dropped_not_guessed_at():
    reply = json.dumps({"facts": [{"fact": "No quote here."}, "not a fact", {"fact": "", "quote": F1[1]},
                                  {"fact": "x" * (module.MAX_SENTENCE_CHARS + 1), "quote": F1[1]},
                                  {"fact": F1[0], "quote": "m" * (module.MAX_QUOTE_CHARS + 1)},
                                  {"fact": F1[0], "quote": F1[1]}]})
    model = FakeChat([reply, answer(("Moved.", ["f1"]))])
    result = await synthesize_passages(model, QUESTION, [P1], mode="facts")
    assert result.notes_dropped_malformed == 5 and result.notes_kept == 1
    assert result.rounds[0].notes_returned == 6 and result.rounds[0].notes_kept == 1


async def test_a_reply_that_is_not_a_fact_list_is_unreadable_and_the_next_passage_is_still_read():
    model = FakeChat([json.dumps({"notes": []}), facts(F2), answer(("Audit errors.", ["f1"]))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="facts")
    assert [r.status for r in result.rounds] == ["unreadable", "noted"]
    assert [s.text for s in result.sentences] == ["Audit errors."] and result.status == "partial"
    assert result.passages_read == 1 and result.passages_unread == 1
    assert result.reasons == ("round 1: the reply could not be read as facts",)


async def test_an_answer_sentence_citing_no_fact_is_dropped_and_counted():
    written = json.dumps({"answer": [{"sentence": "Kept.", "facts": ["f2"]}, {"sentence": "Unknown fact.", "facts": ["f7"]},
                                     {"sentence": "Nothing.", "facts": []}, "not a sentence"]})
    model = FakeChat([facts(F1), facts(F2), written])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="facts")
    assert [s.text for s in result.sentences] == ["Kept."]
    assert result.fold_dropped_uncited == 3 and result.status == "synthesized"
    assert result.record()["facts"] == {"extracted": 2, "used": 1, "unsent": 0, "sentences_dropped_uncited": 3}


async def test_facts_used_counts_distinct_facts_the_shown_sentences_cite_not_distinct_quotes():
    same_quote = ("Priya was the one who moved it.", F1[1])
    replies = [facts(F1, same_quote), facts(F3),
               answer(("It moved.", ["f1"]), ("Priya moved it.", ["f2", "f1"]), ("A party is booked.", ["f3", "f3"]))]
    result = await synthesize_passages(FakeChat(list(replies)), QUESTION, [P1, P3], mode="facts")
    assert len({c for s in result.sentences for c in s.citations}) == 2, "two of the facts share one citation"
    assert result.facts_used == 3 and result.record()["facts"]["used"] == 3
    model = FakeChat(list(replies))
    cut = await synthesize_passages(model, QUESTION, [P1, P3], mode="facts", limits=SynthesisLimits(max_sentences=1))
    assert model.calls[2][1].endswith("Answer in at most 1 sentence(s).")
    assert [s.text for s in cut.sentences] == ["It moved."] and cut.truncated
    assert cut.facts_used == 1, "a sentence cut by the bound uses no fact"


async def test_one_fact_is_still_written_into_an_answer():
    model = FakeChat([facts(F1), answer(("The launch is in March now.", ["f1"]))])
    result = await synthesize_passages(model, QUESTION, [P1], mode="facts")
    assert result.model_calls == 2 and result.folded is True
    assert [s.text for s in result.sentences] == ["The launch is in March now."]


async def test_no_fact_means_no_answer_call_and_no_evidence():
    model = FakeChat([facts(), facts(("Wrong.", "not in the passage"))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="facts")
    assert result.model_calls == 2 and len(model.calls) == 2
    assert result.status == "no_evidence" and result.sentences == () and result.folded is False
    assert result.reasons == (), "no facts is not an answer that failed to fit or be read"
    assert result.record()["facts"] == {"extracted": 0, "used": 0, "unsent": 0, "sentences_dropped_uncited": 0}


async def test_an_answer_that_cannot_be_read_shows_the_facts_unmerged_and_says_so():
    model = FakeChat([facts(F1), facts(F2), "I would rather not answer in JSON."])
    result = await synthesize_passages(model, QUESTION, [P1, P2], mode="facts")
    assert result.folded is False and result.status == "partial"
    assert [s.text for s in result.sentences] == [F1[0], F2[0]]
    assert [s.citations[0].quote for s in result.sentences] == [F1[1], F2[1]]
    assert result.reasons == ("answer: the reply could not be read as an answer; facts shown unmerged",)
    assert result.facts_used == 2


async def test_an_answer_whose_every_sentence_cites_nothing_shows_the_facts_unmerged():
    model = FakeChat([facts(F1), answer(("Nothing cited.", []))])
    result = await synthesize_passages(model, QUESTION, [P1], mode="facts")
    assert [s.text for s in result.sentences] == [F1[0]] and result.status == "partial"
    assert result.fold_dropped_uncited == 1
    assert result.reasons == ("answer: no sentence cited a known fact; facts shown unmerged",)


async def test_a_failing_answer_call_shows_the_facts_unmerged():
    model = FakeChat([facts(F1), ChatError("down")])
    result = await synthesize_passages(model, QUESTION, [P1], mode="facts")
    assert [s.text for s in result.sentences] == [F1[0]] and result.status == "partial"
    assert result.reasons == ("answer model failed: ChatError; facts shown unmerged",)


async def test_extraction_calls_are_bounded_by_rounds_and_the_answer_is_still_written():
    model = FakeChat([facts(F1), facts(F2), answer(("Moved and audited.", ["f1", "f2"]))])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=SynthesisLimits(max_rounds=2), mode="facts")
    assert result.model_calls == 3 and result.passages_read == 2 and result.passages_unread == 1
    assert result.truncated and result.status == "partial" and result.folded is True
    assert "1 passage(s) unread: the bound of 2 round(s) was reached" in result.reasons


async def test_the_answer_call_holds_facts_inside_the_round_bound_and_counts_those_left_out():
    limits = SynthesisLimits(max_round_bytes=70)  # every passage fits; one fact line does, two do not
    assert len(line(1, F1).encode()) <= 70 < len(f"{line(1, F1)}\n{line(2, F2)}".encode())
    model = FakeChat([facts(F1), facts(F2), answer(("Moved.", ["f1"]), ("Audited.", ["f2"]))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=limits, mode="facts")
    written_from = model.calls[2][1]
    assert line(1, F1) in written_from and F2[0] not in written_from
    assert [s.text for s in result.sentences] == ["Moved."], "a fact that was not sent cannot be cited"
    assert result.fold_dropped_uncited == 1 and result.facts_unsent == 1
    assert result.truncated is True and result.status == "partial"
    assert "answer: 1 fact(s) left out: the facts past 70 bytes did not fit the answer's round" in result.reasons
    assert result.record()["facts"] == {"extracted": 2, "used": 1, "unsent": 1, "sentences_dropped_uncited": 1}


async def test_when_no_fact_fits_the_answer_round_no_answer_is_asked_for_and_the_facts_are_shown():
    long = ("The launch, " + "which everyone had been waiting on, " * 3 + "moved to March.", "moved the launch")
    model = FakeChat([facts(long)])
    result = await synthesize_passages(model, QUESTION, [P1], limits=SynthesisLimits(max_round_bytes=70), mode="facts")
    assert len(model.calls) == 1 and result.model_calls == 1, "no answer call is sent over the bound"
    assert [s.text for s in result.sentences] == [long[0]] and result.folded is False
    assert result.facts_unsent == 1 and result.truncated and result.status == "partial"
    assert result.reasons == ("answer: 1 fact(s) left out: the facts past 70 bytes did not fit the answer's round",
                              "answer: no fact fit the answer's round; facts shown unmerged")


async def test_other_modes_record_no_facts():
    model = FakeChat([json.dumps({"notes": [{"sentence": "Moved.", "passage": "chunk:1", "quote": F1[1]}]})])
    result = await synthesize_passages(model, QUESTION, [P1])
    assert result.record()["facts"] is None, "a mode that extracts no facts must not read as having used none"
    assert result.facts_used == 0 and result.facts_unsent == 0


def test_the_answer_rounds_bytes_count_the_newlines_between_facts_and_every_byte_of_them():
    assert module._lines_within(["ab", "cd"], 4) == 1, "two 2-byte lines and a newline are 5 bytes"
    assert module._lines_within(["ab", "cd"], 5) == 2
    assert module._lines_within(["ab"], 2) == 1, "a line of exactly the room fits"
    assert module._lines_within(["\u00e9"], 1) == 0, "bytes, not characters"
    assert module._lines_within([], 64) == 0
