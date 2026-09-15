"""Which passage each sentence of an answer came from, found without a model.

An answer composed from recalled passages reads as sourced whether or not
it is. The reference asks the model to cite numbered sources and trusts
the numbers. Here an answer already written is aligned to the passages
it was given, mechanically: a sentence sharing a run of words with a
passage is quoted, one whose content words mostly appear in one passage
overlaps it, and the rest are unattributed. Numbers in a sentence that
its passage does not hold are named. Word overlap is not support, and the
record says so: it never claims the sentence is true.
"""

from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.attribution import (MAX_ANSWER_CHARS, MAX_PASSAGES, OVERLAP_SHARE, QUOTE_WORDS,
                                                attribute_answer)
from scone_memory.retrieval.synthesis import Passage

P1 = Passage("chunk:1", "Priya moved the launch to March because the audit ran late. The venue stayed the same.")
P2 = Passage("chunk:2", "Tomas said the audit found two billing errors in February, both fixed within a week.")
P3 = Passage("chunk:3", "The launch party is booked for the twelfth of March at the harbour hall.")


def test_a_sentence_sharing_a_run_of_words_is_quoted_with_the_span_it_matched():
    answer = "Priya moved the launch to March because the audit ran late."
    [sentence] = attribute_answer(answer, [P1, P2, P3]).sentences
    assert sentence.status == "quoted" and sentence.passage == "chunk:1"
    assert sentence.quote is not None
    start, end = sentence.quote
    assert P1.text[start:end] == "Priya moved the launch to March because the audit ran late"
    assert (sentence.start, sentence.end) == (0, len(answer))


def test_a_run_is_matched_whatever_the_case_and_punctuation_between_words():
    [sentence] = attribute_answer("the AUDIT found two billing-errors in February.", [P1, P2]).sentences
    assert sentence.status == "quoted" and sentence.passage == "chunk:2"
    assert P2.text[slice(*sentence.quote)] == "the audit found two billing errors in February"


def test_a_sentence_whose_content_words_mostly_appear_in_one_passage_overlaps_it():
    [sentence] = attribute_answer("In February two errors in billing were found by the audit.", [P1, P2, P3]).sentences
    assert sentence.status == "overlapping" and sentence.passage == "chunk:2" and sentence.quote is None
    assert sentence.coverage >= OVERLAP_SHARE and sentence.shared >= 4


def test_a_sentence_no_passage_bears_out_is_unattributed():
    # "audit" is in two passages: one word of six is some overlap, and well short of the rule.
    [sentence] = attribute_answer("The chief executive resigned over the audit delay.", [P1, P2, P3]).sentences
    assert sentence.status == "unattributed" and sentence.passage is None and sentence.quote is None


def test_numbers_the_passage_does_not_hold_are_named():
    [sentence] = attribute_answer("The launch party is booked for the 14th of March at the harbour hall.",
                                  [P3]).sentences
    assert sentence.passage == "chunk:3" and sentence.numbers_missing == ("14th",)
    [clean] = attribute_answer("Tomas said the audit found two billing errors in February.", [P2]).sentences
    assert clean.numbers_missing == ()


def test_each_sentence_is_attributed_on_its_own_and_counted():
    answer = ("Priya moved the launch to March because the audit ran late. "
              "The party is at the harbour hall on the twelfth of March. "
              "Nobody asked the board. Yes.")
    made = attribute_answer(answer, [P1, P2, P3])
    assert [s.status for s in made.sentences] == ["quoted", "overlapping", "unattributed", "too_short"]
    assert [s.passage for s in made.sentences] == ["chunk:1", "chunk:3", None, None]
    assert [answer[s.start:s.end] for s in made.sentences][2] == "Nobody asked the board."
    assert made.counts() == {"quoted": 1, "overlapping": 1, "unattributed": 1, "too_short": 1}


def test_the_longest_run_wins_between_passages():
    # Both are quotes: six words shared with this one, eleven with the other.
    near = Passage("chunk:9", "Priya moved the launch to March.")
    [sentence] = attribute_answer("Priya moved the launch to March because the audit ran late.", [near, P1]).sentences
    assert sentence.passage == "chunk:1"


def test_a_quote_beats_a_passage_holding_more_of_the_words_in_another_order():
    quoting = Passage("chunk:7", "Priya moved the launch to March.")
    scrambled = Passage("chunk:8", "Late, the audit: because it ran, March moved the launch for Priya.")
    [sentence] = attribute_answer("Priya moved the launch to March because the audit ran late.",
                                  [scrambled, quoting]).sentences
    assert sentence.status == "quoted" and sentence.passage == "chunk:7", sentence


def test_the_record_states_its_rules_and_never_claims_accuracy():
    record = attribute_answer("Priya moved the launch to March because the audit ran late.", [P1]).record()
    assert record["verified_accuracy"] is False and record["schema_version"] == 1
    assert record["rules"] == {"quote_words": QUOTE_WORDS, "overlap_share": OVERLAP_SHARE, "measured": False}
    assert record["counts"]["quoted"] == 1
    assert record["sentences"][0]["quote"] == {"start": 0, "end": 58,
                                               "text": "Priya moved the launch to March because the audit ran late"}


def test_limits_are_refused_not_cut():
    with pytest.raises(InvalidInput):
        attribute_answer("x" * (MAX_ANSWER_CHARS + 1), [P1])
    with pytest.raises(InvalidInput):
        attribute_answer("An answer.", [Passage(f"chunk:{n}", "text") for n in range(MAX_PASSAGES + 1)])
    with pytest.raises(InvalidInput):
        attribute_answer("An answer.", [P1, Passage("chunk:1", "the same id twice")])
    with pytest.raises(InvalidInput):
        attribute_answer("   ", [P1])


def test_a_run_shorter_than_a_quote_counts_for_nothing_against_more_words():
    # This one shares a three-word run and little else; the other shares no run and most words.
    runs = Passage("chunk:5", "Nobody expected the audit ran over budget that year.")
    words = Passage("chunk:6", "February: billing errors, two of them, turned up during the audit.")
    [sentence] = attribute_answer("In February the audit ran into two billing errors.", [runs, words]).sentences
    assert sentence.status == "overlapping" and sentence.passage == "chunk:6", sentence
