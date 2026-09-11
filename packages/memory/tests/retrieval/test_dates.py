"""Dates a question refers to, resolved against the moment it is asked.

Real memory questions are anchored in time: "in May 2023", "three weeks
ago", "last month". Each reference becomes a window of whole days, the
first day in and the first day out, carrying the words that made it, so
a caller can say what it understood.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from scone_memory.retrieval.dates import DateWindow, date_windows

NOW = "2023-04-20T10:12:00Z"  # a Thursday


def spans(question: str, now: str = NOW) -> list[tuple[str, str, str]]:
    return [(w.start.isoformat(), w.end.isoformat(), w.words) for w in date_windows(question, now=now)]


@pytest.mark.parametrize("question, start, words", [
    ("What did I do on 2023-05-20?", "2023-05-20", "2023-05-20"),
    ("what happened on 2023/05/20", "2023-05-20", "2023/05/20"),
    ("Where was I on May 20, 2023?", "2023-05-20", "may 20, 2023"),
    ("Where was I on May 20th 2023?", "2023-05-20", "may 20th 2023"),
    ("Where was I on 20 May 2023?", "2023-05-20", "20 may 2023"),
    ("and on the 3rd of June, 2022?", "2022-06-03", "3rd of june, 2022"),
])
def test_a_named_day_is_that_day(question, start, words):
    [(first, end, said)] = spans(question)
    assert first == start and said == words
    assert date.fromisoformat(end) == date.fromisoformat(start) + timedelta(days=1)


def test_a_month_and_a_year_are_the_whole_of_them():
    assert spans("What did I buy in May 2023?") == [("2023-05-01", "2023-06-01", "may 2023")]
    assert spans("Who did I meet in Feb. 2024?") == [("2024-02-01", "2024-03-01", "feb. 2024")]
    assert spans("What changed in 2021?") == [("2021-01-01", "2022-01-01", "2021")]


def test_a_month_without_a_year_is_the_latest_one_begun():
    assert spans("What did I cook in March?") == [("2023-03-01", "2023-04-01", "march")]
    assert spans("What did I cook in April?") == [("2023-04-01", "2023-05-01", "april")]
    assert spans("What did I cook in May?") == [("2022-05-01", "2022-06-01", "may")]


@pytest.mark.parametrize("question, start, end", [
    ("What did I eat yesterday?", "2023-04-19", "2023-04-20"),
    ("What have I done today?", "2023-04-20", "2023-04-21"),
    ("Who did I meet 9 days ago?", "2023-04-11", "2023-04-12"),
    ("Who did I meet nine days ago?", "2023-04-11", "2023-04-12"),
    ("What did I do a day ago?", "2023-04-19", "2023-04-20"),
    ("What did I read two weeks ago?", "2023-04-03", "2023-04-10"),
    ("What did I read a week ago?", "2023-04-10", "2023-04-17"),
    ("Where did I go 2 months ago?", "2023-02-01", "2023-03-01"),
    ("Where did I live a year ago?", "2022-01-01", "2023-01-01"),
])
def test_ago_is_that_many_days_weeks_months_or_years_back(question, start, end):
    [(first, last, _)] = spans(question)
    assert (first, last) == (start, end)


@pytest.mark.parametrize("question, start, end", [
    ("What did I plan last week?", "2023-04-10", "2023-04-17"),
    ("What did I plan last month?", "2023-03-01", "2023-04-01"),
    ("What did I plan last year?", "2022-01-01", "2023-01-01"),
    ("What have I planned this week?", "2023-04-17", "2023-04-24"),
    ("What have I planned this month?", "2023-04-01", "2023-05-01"),
    ("What have I planned this year?", "2023-01-01", "2024-01-01"),
])
def test_last_and_this_are_whole_calendar_units(question, start, end):
    [(first, last, _)] = spans(question)
    assert (first, last) == (start, end)


@pytest.mark.parametrize("question, start", [
    ("Which concerts did I attend in the past two months?", "2023-02-20"),
    ("What did I buy in the last 10 days?", "2023-04-10"),
    ("What did I watch over the past 3 weeks?", "2023-03-30"),
])
def test_the_past_few_units_run_up_to_today(question, start):
    [(first, last, _)] = spans(question)
    assert (first, last) == (start, "2023-04-21")


def test_a_day_a_month_back_that_the_month_lacks_is_its_last():
    assert spans("What did I spend in the past month?", now="2023-03-31T09:00:00Z") == [
        ("2023-02-28", "2023-04-01", "the past month")]
    assert spans("And over the last month?", now="2023-03-31T09:00:00Z") == [
        ("2023-02-28", "2023-04-01", "the last month")]


def test_a_weekday_is_the_latest_one_before_today():
    assert spans("Who did I call last Tuesday?") == [("2023-04-18", "2023-04-19", "last tuesday")]
    assert spans("Who did I call on Thursday?") == [("2023-04-13", "2023-04-14", "thursday")]


def test_the_longest_reading_wins_and_each_is_kept_in_order():
    assert [w for *_, w in spans("Between May 20, 2023 and June 2023, and yesterday?")] == [
        "may 20, 2023", "june 2023", "yesterday"]


@pytest.mark.parametrize("question", [
    "How many rooms are in the house?",
    "I may go tomorrow",  # may the verb, and a future day nothing has happened on
    "Room 2021 is booked",  # a number, not a year
    "What did I do 0 days ago?",
    "on 2023-02-30",  # no such day
    "I rewatched Friday Night Lights",  # a title, not a day
])
def test_what_is_not_a_date_is_left_alone(question):
    assert date_windows(question, now=NOW) == []


def test_a_window_says_its_first_day_in_and_first_day_out():
    [window] = date_windows("yesterday", now=NOW)
    assert window == DateWindow(date(2023, 4, 19), date(2023, 4, 20), "yesterday")
    assert window.holds("2023-04-19T23:59:59Z") and not window.holds("2023-04-20T00:00:00Z")
