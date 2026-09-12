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
    questions: int = 0
    routes: dict[str, int] = field(default_factory=dict)
    #: Questions the rule sent to the computer, and how many of those the
    #: file's own answer agrees with.
    computed: int = 0
    computed_right: int = 0

    def record(self) -> dict[str, object]:
        return {"dataset": self.dataset, "questions": self.questions, "routes": dict(self.routes),
                "computed": self.computed, "computed_right": self.computed_right}

    def text(self) -> str:
        went = ", ".join(f"{name} {self.routes.get(name, 0)}" for name in ROUTES)
        scored = (f"; of {self.computed} computed, {self.computed_right} agree with the file"
                  if self.computed else "")
        return (f"routing: {self.questions} question(s) of {self.dataset}: {went}{scored}. "
                f"This does not say whether a question sent to search would have been answered "
                f"better another way: the file has one answer, not one per route.")


async def run_route_bench(dataset: str | Path, *, limit: Optional[int] = None) -> RouteScore:
    """Route every question in the file, each on its own memory."""
    items = load_items(dataset)[:limit]
    routes = {name: 0 for name in ROUTES}
    computed = right = 0
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
        computed += 1
        detail = answered.detail
        value = detail.get("value") if isinstance(detail, dict) else None
        right += 1 if score_answer(value if isinstance(value, dict) else {}, None,
                                   expected.get(item.question_id, "")) else 0
    return RouteScore(dataset=str(dataset), questions=len(items), routes=routes,
                      computed=computed, computed_right=right)
