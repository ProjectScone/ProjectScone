"""Reading a temporal question as an operator over events.

A quarter of what people ask memory is arithmetic over dates: how long
between two things, how long ago something was, which of them came
first. Retrieval is good at finding which passage an event phrase means;
software is exact at subtracting dates. The planner splits the work
along that line, and refuses anything it cannot read confidently: a
computed answer arrives stated as fact, so a wrong one is worse than
none.
"""

from __future__ import annotations

import pytest

from scone_memory.retrieval.temporal import plan


def test_how_many_units_between_two_events():
    asked = plan("How many weeks passed between the time I sold baked goods at the market "
                 "and the time I ran the charity bake-off?")
    assert asked.kind == "between" and asked.unit == "week"
    assert asked.events == ("i sold baked goods at the market", "i ran the charity bake-off")


@pytest.mark.parametrize("question, events", [
    ("How many days after I started the course did I finish it?", ("i started the course", "i finish it")),
    ("How many months from my move to Lisbon to my first day at Acme?",
     ("my move to lisbon", "my first day at acme")),
    ("How long was it between my trip to Rome and my trip to Paris?", ("my trip to rome", "my trip to paris")),
])
def test_the_two_events_are_taken_whole(question, events):
    asked = plan(question)
    assert asked.kind == "between" and asked.events == events


def test_how_long_ago_an_event_was():
    assert plan("How many days ago did I meet Emma?") == plan("How many days ago did I meet Emma")
    asked = plan("How many days ago did I meet Emma?")
    assert (asked.kind, asked.unit, asked.events) == ("since", "day", ("meet emma",))
    loose = plan("How long ago did I attend the summer nights festival?")
    assert (loose.kind, loose.unit, loose.events) == ("since", None, ("attend the summer nights festival",))
    since = plan("How many months has it been since I quit my job?")
    assert (since.kind, since.unit, since.events) == ("since", "month", ("i quit my job",))


@pytest.mark.parametrize("question, kind", [
    ("Which event happened first, my post about chili or my plank challenge?", "first"),
    ("Which trip did I take first, the one to Europe or the solo trip to Thailand?", "first"),
    ("Which came earlier: my move to Porto or my job at Acme?", "first"),
    ("Which of them happened last, my move to Porto or my job at Acme?", "last"),
    ("Which did I do most recently, my move to Porto or my job at Acme?", "last"),
])
def test_which_of_two_came_first_or_last(question, kind):
    asked = plan(question)
    assert asked.kind == kind and len(asked.events) == 2 and all(" " in event for event in asked.events)


def test_three_events_put_in_order():
    asked = plan("Which three events happened in the order from first to last: the day I helped with the "
                 "nursery, the day I picked the cake, and the day I met Emma?")
    assert asked.kind == "order"
    assert asked.events == ("i helped with the nursery", "i picked the cake", "i met emma")


def test_the_events_of_an_ordering_question_may_be_quoted():
    asked = plan("In what order did these happen: 'the summer nights festival', 'the plank challenge'?")
    assert asked.kind == "order" and asked.events == ("the summer nights festival", "the plank challenge")


@pytest.mark.parametrize("question", [
    "What did I eat yesterday?",
    "How many books did I read last year?",  # a count of things, not of days
    "How old was I when I moved to the United States?",  # an age needs a birth date, not two events
    "What is the order of the six museums I visited?",  # the events are not named
    "Which event happened first?",  # neither event is named
    "How many days are in a fortnight?",
    "How long is the Amazon?",
    "",
])
def test_what_cannot_be_read_confidently_is_not_planned(question):
    assert plan(question) is None


def test_a_plan_says_what_it_read_for_the_answer_to_show():
    asked = plan("How many weeks between my trip to Rome and my trip to Paris?")
    assert asked.record() == {"kind": "between", "unit": "week",
                              "events": ["my trip to rome", "my trip to paris"]}


def test_the_last_and_splits_a_pair_so_either_half_may_hold_its_own():
    asked = plan("How many weeks between the time I met my aunt and received the chandelier "
                 "and the time I ran the charity bake-off?")
    assert asked.events == ("i met my aunt and received the chandelier", "i ran the charity bake-off")


def test_an_event_a_question_does_not_name_is_not_planned():
    """One word is a thing, not an event this can ground confidently."""
    assert plan("How many days between Rome and Paris?") is None


def test_since_one_event_when_another_is_the_span_between_them():
    asked = plan("How many weeks had passed since I recovered from the flu when I went on my 10th jog outdoors?")
    assert asked.kind == "between" and asked.unit == "week"
    assert asked.events == ("i recovered from the flu", "i went on my 10th jog outdoors")


@pytest.mark.parametrize("question, words", [
    ("What did I do 5 days ago?", "5 days ago"),
    ("Who did I meet with during the lunch last Tuesday?", "last tuesday"),
    ("What charity event did I participate in a month ago?", "a month ago"),
    ("I received a piece of jewelry last Saturday from whom?", "last saturday"),
])
def test_a_question_about_one_day_asks_what_was_recorded_then(question, words):
    asked = plan(question, now="2023-04-20T10:12:00Z")
    assert asked.kind == "on" and asked.window is not None and asked.window.words == words
    assert asked.events and "ago" not in asked.events[0] and "?" not in asked.events[0]


@pytest.mark.parametrize("question", [
    "How many books did I read last year?",  # a count of things, not what happened
    "How long was I away last month?",  # a length of time, and no event to date
    "How old was I in 2019?",
    "What did I do with Rachel on the Wednesday two months ago?",  # two readings of one day
    "I may go tomorrow",  # nothing recorded ahead of the moment asked
])
def test_a_dated_question_that_is_not_a_lookup_is_not_planned_as_one(question):
    asked = plan(question, now="2023-04-20T10:12:00Z")
    assert asked is None or asked.kind != "on"


@pytest.mark.parametrize("question, kind, event", [
    ("How long did Alice work at Acme?", "held", "alice work at acme"),
    ("How long has Alice worked at Acme Robotics?", "held", "alice worked at acme robotics"),
    ("How long was the project open?", "held", "project open"),
    ("When did Alice join Acme?", "when", "alice join acme"),
    ("When was Alice the manager of the Lisbon office?", "when", "alice the manager of the lisbon office"),
])
def test_a_question_about_a_claim_asks_the_ledger_not_the_passages(question, kind, event):
    """How long something held, and when it began or ended, are the valid
    time of a claim, which the ledger keeps exactly."""
    asked = plan(question, now="2023-04-20T10:12:00Z")
    assert (asked.kind, asked.events) == (kind, (event,))


@pytest.mark.parametrize("question", [
    "How long ago did I meet Emma?",  # a distance from now, not the length of a claim
    "How long was it between my trip to Rome and my trip to Paris?",  # two events
    "How long is the Amazon?",
])
def test_a_length_that_is_not_a_claims_own_is_not_asked_of_the_ledger(question):
    asked = plan(question, now="2023-04-20T10:12:00Z")
    assert asked is None or asked.kind in ("since", "between")


def test_a_question_naming_a_day_is_not_a_question_about_a_claim():
    """"When did I meet Emma five days ago" names the day; the ledger is
    asked only when nothing else dates the question."""
    asked = plan("When did I meet Emma 5 days ago?", now="2023-04-20T10:12:00Z")
    assert asked.kind == "on" and asked.window is not None and asked.window.words == "5 days ago"
