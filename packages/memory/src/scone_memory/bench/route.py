"""Measuring the rule that chooses how a question is answered.

A routing rule written down is better than one a model invents, but only
if somebody checks it. This runs the rule over a file of questions whose
answers are known, reports where each one went, and scores the ones it
sent to the computer against the file's own answers.

What it cannot say is whether a question that went to search would have
been answered better by another route. That needs an answer to compare
against for every question under every route, which these files do not
have, so the report says what it does not know rather than reading as
though the rule had been vindicated.

Nothing here calls a model: the same hash embedder, one memory per
question, and the file's own answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

from ..retrieval.router import ROUTES, answer_question
from .runner import iso_date, load_items
from .temporal import _engine_for, score_answer


@dataclass(frozen=True)
class RouteScore:
    """Where the questions went, and how the computed ones fared."""

    dataset: str
    #: Questions read. With a limit this is fewer than the file holds, and
    #: `questions_found` is what it holds.
    questions: int = 0
    questions_found: int = 0
    routes: dict[str, int] = field(default_factory=dict)
    #: Questions the temporal route actually **computed** an answer for.
    #: The route answers two ways — it computes, or it hands back the day's
    #: passages — and only the first has arithmetic to score.
    computed: int = 0
    #: Questions the temporal route answered from passages instead. No
    #: arithmetic in them, so nothing here can be scored; counting them in
    #: `computed` made the ratio report the computation as wrong when
    #: nothing had been computed.
    recalled: int = 0
    #: Of the computed ones: agreed with the file, disagreed with it, and
    #: came back in a shape the scorer cannot judge. Three facts, because
    #: adding "cannot tell" to "wrong" tells a reader neither.
    computed_right: int = 0
    computed_wrong: int = 0
    computed_unscored: int = 0

    def record(self) -> dict[str, object]:
        return {"dataset": self.dataset, "questions": self.questions,
                "questions_found": self.questions_found, "capped": self.capped,
                "routes": dict(self.routes), "computed": self.computed,
                "recalled": self.recalled, "computed_right": self.computed_right,
                "computed_wrong": self.computed_wrong,
                "computed_unscored": self.computed_unscored}

    @property
    def capped(self) -> bool:
        return self.questions < self.questions_found

    def text(self) -> str:
        went = ", ".join(f"{name} {self.routes.get(name, 0)}" for name in ROUTES)
        read = (f"{self.questions} of {self.questions_found} question(s)" if self.capped
                else f"{self.questions} question(s)")
        scored = ""
        if self.computed:
            scored = (f"; of {self.computed} computed, {self.computed_right} agree with the file, "
                      f"{self.computed_wrong} disagree")
            if self.computed_unscored:
                scored += f", {self.computed_unscored} in a shape this cannot judge"
        if self.recalled:
            scored += (f"; {self.recalled} more went to the temporal route and were read from "
                       f"passages rather than computed, so there is no arithmetic to score")
        return (f"routing: {read} of {self.dataset}: {went}{scored}. "
                f"This does not say whether a question sent to search would have been answered "
                f"better another way: the file has one answer, not one per route.")


async def run_route_bench(dataset: str | Path, *, limit: Optional[int] = None) -> RouteScore:
    """Route every question in the file, each on its own memory."""
    whole = load_items(dataset)
    items = whole[:limit]
    routes = {name: 0 for name in ROUTES}
    computed = recalled = right = wrong = unscored = 0
    # The loader keeps only what retrieval is scored on, so the answers are
    # read from the file itself, as the temporal bench reads them. Without
    # this the computed score would be a number that looks measured and is
    # always zero.
    expected = {str(raw.get("question_id", "")): str(raw.get("answer", ""))
                for raw in json.loads(Path(dataset).read_text(encoding="utf-8"))}
    for item in items:
        engine, _ = await _engine_for(item)
        try:
            answered = await answer_question(engine, "default", item.question,
                                             now=iso_date(item.question_date))
        finally:
            await engine.close()
        routes[answered.route] = routes.get(answered.route, 0) + 1
        if answered.route != "temporal":
            continue
        detail = answered.detail if isinstance(answered.detail, dict) else {}
        # The temporal route computes or falls back to the day's passages,
        # and says which. Only the first has arithmetic to compare.
        if str(detail.get("status", "")) != "computed":
            recalled += 1
            continue
        computed += 1
        value = detail.get("value")
        agreed = score_answer(value if isinstance(value, dict) else {}, None,
                              expected.get(item.question_id, ""))
        if agreed is None:
            unscored += 1
        elif agreed:
            right += 1
        else:
            wrong += 1
    return RouteScore(dataset=str(dataset), questions=len(items), questions_found=len(whole),
                      routes=routes, computed=computed, recalled=recalled, computed_right=right,
                      computed_wrong=wrong, computed_unscored=unscored)
