"""Does splitting a multi-part question retrieve better? Measured, paired.

Searching each part of a question separately is obviously appealing and
not obviously useful, so this asks the same questions both ways on the
same memory and reports the difference **in questions**, not in
percentages: a percentage over a handful of questions reads as a result
and is noise.

The honest headline is the split count. The rule that splits questions is
deliberately shy, so on a given file it may split very few — and if it
splits none, it changes nothing here and the report says exactly that
rather than reporting two identical numbers as agreement.

Only the split questions are compared. On a question the rule leaves
whole, parted recall *is* the ordinary search, so including those would
pad both columns with identical rows and make a small difference look
smaller.

Nothing here calls a model: the hash embedder, one memory per question.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..retrieval.decompose import decompose
from ..retrieval.parts import recall_parts
from .runner import BenchItem, load_items
from .temporal import _engine_for


@dataclass(frozen=True)
class PartsScore:
    """What splitting changed, on the questions it applied to."""

    dataset: str
    k: int
    #: Questions read. With a limit this is fewer than the file holds, and
    #: `questions_found` is what it holds -- the report prints this number
    #: as "of <dataset>", so on its own it would understate the corpus.
    questions: int = 0
    questions_found: int = 0
    #: Questions the rule split. The rest are untouched by the feature.
    split: int = 0
    #: On the split questions only: a returned passage from any session
    #: the answer rests on, and from every such session.
    whole_any: int = 0
    parted_any: int = 0
    whole_all: int = 0
    parted_all: int = 0
    #: Split questions where a part's own search found nothing.
    parts_unanswered: int = 0

    @property
    def capped(self) -> bool:
        return self.questions < self.questions_found

    def record(self) -> dict[str, object]:
        return {"dataset": self.dataset, "k": self.k, "questions": self.questions,
                "questions_found": self.questions_found, "capped": self.capped,
                "split": self.split, "whole_any": self.whole_any, "parted_any": self.parted_any,
                "whole_all": self.whole_all, "parted_all": self.parted_all,
                "parts_unanswered": self.parts_unanswered}

    def text(self) -> str:
        read = (f"{self.questions} of {self.questions_found} question(s)" if self.capped
                else f"{self.questions} question(s)")
        if not self.split:
            return (f"parts: the rule split none of {read} of {self.dataset}, "
                    f"so splitting changes nothing on this file. Nothing is measured by comparing "
                    f"a question with itself.")
        return (f"parts: {self.split} of {read} of {self.dataset} split at k="
                f"{self.k}. On those: any-evidence {self.whole_any} whole vs {self.parted_any} "
                f"parted; all-evidence {self.whole_all} whole vs {self.parted_all} parted "
                f"({self.parted_all - self.whole_all:+d} question(s)). "
                f"{self.parts_unanswered} had a part its own search could not answer.")


def _sessions(items, told: dict[int, str], k: int) -> set[str]:
    return {told.get(item.episode_id, "") for item in items[:k]}


def _scored(item: BenchItem, seen: set[str]) -> tuple[bool, bool]:
    wanted = [sid for sid in item.answer_session_ids if sid]
    return (any(sid in seen for sid in wanted),
            bool(wanted) and all(sid in seen for sid in wanted))


async def run_parts_bench(dataset: str | Path, *, limit: Optional[int] = None,
                          k: int = 10) -> PartsScore:
    """Ask every question both ways on its own memory, and count."""
    every = load_items(dataset)
    items = every[:limit]
    split = whole_any = parted_any = whole_all = parted_all = unanswered = 0
    for item in items:
        if not decompose(item.question).split:
            continue
        split += 1
        engine, told = await _engine_for(item)
        try:
            whole = await engine.recall("default", item.question, limit=k)
            parted = await recall_parts(engine, "default", item.question, limit=k)
        finally:
            await engine.close()
        one, both = _scored(item, _sessions(whole.items, told, k))
        whole_any += one
        whole_all += both
        one, both = _scored(item, _sessions(list(parted.items), told, k))
        parted_any += one
        parted_all += both
        unanswered += bool(parted.unanswered)
    return PartsScore(dataset=str(dataset), k=k, questions=len(items),
                      questions_found=len(every), split=split,
                      whole_any=whole_any, parted_any=parted_any, whole_all=whole_all,
                      parted_all=parted_all, parts_unanswered=unanswered)
