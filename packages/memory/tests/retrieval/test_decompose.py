"""Splitting a question that asks more than one thing.

The risk runs one way. A question wrongly left whole retrieves what it
retrieves today; a question wrongly split is asked as two queries that
mean nothing ("where can I buy salt", "pepper"), and the answer is worse
than before. So the rule is deliberately shy, and these tests pin both
sides of that: what must split, and what must be left alone.
"""

from __future__ import annotations

import pytest

from scone_memory.retrieval.decompose import MAX_PARTS, decompose


def parts(question):
    return [part.text for part in decompose(question).parts]


def test_two_questions_in_one_message_are_two_questions():
    said = "What did I decide about billing? Who was at the meeting?"
    assert parts(said) == ["What did I decide about billing?", "Who was at the meeting?"]


def test_a_conjunction_splits_when_each_side_asks_its_own_thing():
    said = "What did I decide about billing, and who was at the meeting?"
    assert parts(said) == ["What did I decide about billing", "who was at the meeting?"]


def test_a_list_of_things_is_not_two_questions():
    """This is the failure that would make the feature worse than nothing."""
    assert parts("Where can I buy salt and pepper?") == ["Where can I buy salt and pepper?"]


def test_a_question_about_one_relation_between_two_events_stays_whole():
    """Splitting this would destroy the very thing it asks about."""
    said = "How long after I joined the choir did I buy the bicycle?"
    assert parts(said) == [said]


def test_a_semicolon_splits_when_both_sides_stand_up():
    said = "Where do I work; where does my sister work?"
    assert parts(said) == ["Where do I work", "where does my sister work?"]


def test_the_parts_are_the_question_s_own_words_at_their_own_offsets():
    said = "What did I decide about billing, and who was at the meeting?"
    for part in decompose(said).parts:
        assert said[part.start:part.end] == part.text, "a part must be quotable from the question"


def test_a_question_that_does_not_split_says_so():
    whole = decompose("Where can I buy salt and pepper?")
    assert len(whole.parts) == 1 and not whole.split
    assert "one thing" in whole.why, whole.why


def test_a_question_that_splits_says_why_it_did():
    split = decompose("What did I decide about billing? Who was at the meeting?")
    assert split.split and "2 parts" in split.why, split.why


def test_more_parts_than_we_will_search_are_reported_not_hidden():
    said = " ".join(f"Who was at meeting {n}?" for n in range(MAX_PARTS + 3))
    many = decompose(said)
    assert len(many.parts) == MAX_PARTS
    assert many.parts_found == MAX_PARTS + 3 and many.capped
    assert f"{MAX_PARTS + 3} parts" in many.why and "first" in many.why, many.why


def test_an_empty_question_is_refused_rather_than_silently_searched_as_nothing():
    from scone_memory.core.errors import InvalidInput

    with pytest.raises(InvalidInput):
        decompose("   ")


def test_punctuation_alone_never_becomes_a_part():
    for said in ("What did I decide? ?", "Who came?; ;"):
        assert all(part.text.strip(" ?;,.") for part in decompose(said).parts), said


# These four are real LongMemEval questions. The first three are splits the
# rule made and should not have: the synthetic "salt and pepper" case was
# too easy, because it had no auxiliary verb on the right to fool the
# asking test. Real questions do.

def test_a_conjunction_of_two_nouns_does_not_split_just_because_a_verb_follows():
    said = "How many hours of jogging and yoga did I do last week?"
    assert parts(said) == [said], "'yoga did I do last week' is not a question"


def test_a_list_inside_a_noun_phrase_does_not_split():
    said = "What is the total number of goals and assists I have in the recreational indoor soccer league?"
    assert parts(said) == [said]


def test_between_one_thing_and_another_is_one_relation():
    """Splitting this asks for two dates and throws away the subtraction
    that was the question."""
    said = ("How many days passed between the day I cancelled my FarmFresh subscription "
            "and the day I did my online grocery shopping from Instacart?")
    assert parts(said) == [said]


def test_a_second_interrogative_splits_without_needing_a_comma():
    said = "Where do I work and where does my sister work?"
    assert parts(said) == ["Where do I work", "where does my sister work?"]


def test_a_question_mark_inside_a_quoted_title_is_not_a_boundary():
    """Also real. The title is 'To Adapt or Not to Adapt? Real-Time
    Adaptation for Semantic Segmentation', and its question mark is part
    of the name, not the end of a question."""
    said = ("Can you remind me what was the average improvement in framerate when using the "
            "Hardware-Aware Modular Training (HAMT) agent in the 'To Adapt or Not to Adapt? "
            "Real-Time Adaptation for Semantic Segmentation' submission?")
    assert parts(said) == [said]


def test_a_real_join_still_splits_a_question_that_quotes_a_title():
    said = "What does 'To Adapt or Not to Adapt? Real-Time Adaptation' say, and who wrote it?"
    assert parts(said) == ["What does 'To Adapt or Not to Adapt? Real-Time Adaptation' say",
                           "who wrote it?"]


def test_an_apostrophe_is_not_a_quotation_mark():
    said = "What didn't I finish about billing, and who was at the meeting?"
    assert parts(said) == ["What didn't I finish about billing", "who was at the meeting?"]
