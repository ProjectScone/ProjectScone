"""Scoring computed temporal answers on a file of dated questions.

Each item carries a question, the day it is asked, the dated sessions
memory holds, and the answer a person would accept. Every item gets its
own engine, so one question's memory never answers another's.

The scoring uses no model. A number answer is right when the expected
answer holds that number, in the unit asked for or in days, counting a
day either way: people say both "9 days ago" and "10 days including
today", and the files say so themselves. A chosen event is right when
the expected answer names it rather than the one refused, by how many
of its longer words appear. An order is right when the expected answer
puts the same events in the same order.

What the planner will not read, and what it will not date, is counted
apart from what it got wrong, because refusing to answer and answering
badly are different things, and only one of them misleads.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence, cast

from ..backends import InMemoryDocumentStore, InMemoryVectorIndex
from ..embedders.hash import HashEmbedder
from ..memory.engine import MemoryEngine
from ..retrieval.temporal import temporal_answer
from .runner import BenchItem, iso_date, load_items

_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"[a-z0-9']+")
#: Numbers people write as words in an expected answer.
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
          "ten": 10, "eleven": 11, "twelve": 12}
#: Words long enough to tell two events apart.
_TELLING = 4


@dataclass(frozen=True)
class TemporalScore:
    """What a run computed, and what it refused."""

    items: int = 0
    computed: int = 0
    #: Questions answered with the passages of the day they name.
    recalled: int = 0
    #: Of those, the ones that returned a passage from a session the
    #: expected answer rests on.
    recalled_right: int = 0
    correct: int = 0
    wrong: int = 0
    not_temporal: int = 0
    ungrounded: int = 0
    ambiguous: int = 0
    unscored: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        return {"schema_version": 1, "questions": self.items, "computed": self.computed,
                "recalled": self.recalled, "recalled_right": self.recalled_right, "correct": self.correct,
                "wrong": self.wrong, "not_temporal": self.not_temporal, "ungrounded": self.ungrounded,
                "ambiguous": self.ambiguous,
                "correct_of_computed": round(self.correct / self.computed, 4) if self.computed else None,
                "correct_of_questions": round(self.correct / self.items, 4) if self.items else None,
                "wrong_questions": list(self.unscored)}

    def text(self) -> str:
        share = f"{self.correct / self.computed:.0%}" if self.computed else "no answers"
        return "\n".join([
            f"temporal: {self.items} questions; computed {self.computed} of {self.items}, {share} of them right",
            f"correct: {self.correct}; wrong: {self.wrong}",
            f"recalled: {self.recalled} questions about a day it named, {self.recalled_right} of them returning a "
            f"passage the expected answer rests on",
            f"refused: {self.not_temporal} not read, {self.ungrounded} with an event not in memory, "
            f"{self.ambiguous} with an event's day undecided",
            *(f"wrong: {question}" for question in self.unscored),
        ])


def _numbers(answer: str) -> set[int]:
    """The numbers an expected answer holds, digits or words."""
    said = answer.casefold()
    return ({int(found) for found in _NUMBER.findall(said)}
            | {value for word, value in _WORDS.items() if re.search(rf"\b{word}\b", said)})


def _telling(phrase: str) -> set[str]:
    return {word for word in _WORD.findall(phrase.casefold()) if len(word) >= _TELLING}


def score_answer(value: dict[str, object], events: Optional[Sequence[str]], expected: str) -> Optional[bool]:
    """Whether a computed answer matches the expected one, or None when
    this cannot tell (an answer shape it does not score)."""
    expected = str(expected)
    if "asked" in value and "days" in value:
        asked, days = int(str(value["asked"])), int(str(value["days"]))
        return any(abs(ours - said) <= 1 for ours in (asked, days) for said in _numbers(expected))
    if "order" in value:
        listed = cast(Sequence[Mapping[str, object]], value["order"])
        ordered = [str(item["event"]) for item in listed]
        places = [min((expected.casefold().find(word) for word in _telling(event)
                       if word in expected.casefold()), default=-1) for event in ordered]
        seen = [place for place in places if place >= 0]
        return seen == sorted(seen) and len(seen) >= 2
    chosen = str(value.get("first") or value.get("last") or "")
    if not chosen:
        return None
    held = len(_telling(chosen) & _telling(expected))
    others = [len(_telling(event) & _telling(expected)) for event in events or () if event != chosen]
    return held > max(others, default=0)


async def _engine_for(item: BenchItem) -> tuple[MemoryEngine, dict[int, str]]:
    """One memory per question, and which session each episode came from."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    told: dict[int, str] = {}
    for session, when, session_id in zip(item.sessions, item.session_dates, item.session_ids):
        said = "\n".join(session).strip()
        if said:
            told[(await engine.remember("default", said, created_at=when)).episode_id] = session_id
    return engine, told


async def run_temporal(path: str | Path, *, limit: Optional[int] = None) -> TemporalScore:
    """Score every question in the file, each on its own memory."""
    items = load_items(path)[:limit]
    # The expected answers are the file's own; the loader keeps only what
    # retrieval is scored on, so they are read here.
    expected = {str(raw.get("question_id", "")): str(raw.get("answer", ""))
                for raw in json.loads(Path(path).read_text(encoding="utf-8"))}
    counted = {"computed": 0, "recalled": 0, "recalled_right": 0, "correct": 0, "wrong": 0, "not_temporal": 0,
               "ungrounded": 0, "ambiguous": 0}
    unscored: list[str] = []
    for item in items:
        engine, told = await _engine_for(item)
        try:
            answer = await temporal_answer(engine, "default", item.question, now=iso_date(item.question_date))
        finally:
            await engine.close()
        if answer.status == "recalled":
            counted["recalled"] += 1
            wanted = set(item.answer_session_ids)
            shown = {told.get(int(str(passage["episode_id"]))) for passage in answer.anchors}
            counted["recalled_right"] += 1 if shown & wanted else 0
            continue
        if answer.status != "computed":
            counted[answer.status] += 1
            continue
        counted["computed"] += 1
        right = score_answer(answer.value, answer.asked.events if answer.asked else None,
                             str(expected.get(item.question_id, "")))
        counted["correct" if right else "wrong"] += 1
        if not right:
            unscored.append(item.question)
    return TemporalScore(items=len(items), unscored=tuple(unscored), **counted)
