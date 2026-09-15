"""Recorded feedback replayed into ranking: MRR with the feedback weight off and on.

Each subject of the fixture has one passage that answers it, two ways of
asking it and a paraphrase. Every passage is stored once. One half of the
subjects is judged: each way of asking it is recalled (a day apart) and a
judge marks what came back. Then, with no further feedback, every subject's
questions are asked with the weight off and at each weight measured, on the
same engine and the same recorded judgements:

- ``judged``: the paraphrases of the judged half, which no judgement saw;
- ``unrelated``: every question of the other half, which nobody judged.

Judges: ``kind`` marks the answering passage useful when it came back;
``strict`` also marks not useful every passage shown above it (all five when
it did not come back). The replay runs once per half judged. No model runs:
the in-memory stores and ``HashEmbedder``, engine defaults otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends import InMemoryDocumentStore, InMemoryVectorIndex
from ..core.errors import InvalidInput
from ..embedders.hash import HashEmbedder
from ..memory.engine import MemoryEngine
from ..observability.events import InMemoryEventLog
from ..testing import Clock
from .metrics import reciprocal_rank

SPACE = "feedback-replay"
#: What a judge is shown per question.
SHOWN = 5
#: How deep the evaluated rank looks: MRR@10.
DEPTH = 10
#: The weights measured. 0.0001 was chosen on the replay judging half a (the largest whose
#: unrelated MRR stayed within 0.01 of off); half b is its held-out check.
WEIGHTS = (0.00005, 0.0001, 0.0002, 0.0005)
CHOSEN = 0.0001
JUDGES = ("kind", "strict")
HALVES = ("a", "b")
START = "2026-05-01T09:00:00.000Z"


@dataclass
class Replay:
    judged_half: str
    judge: str
    judgements: int
    #: weight -> set ("judged", "unrelated") -> reciprocal ranks, in question order.
    ranks: dict[float, dict[str, list[float]]] = field(default_factory=dict)
    #: weight -> questions whose rank rose or fell against weight 0, as "set:subject:question".
    rose: dict[float, list[str]] = field(default_factory=dict)
    fell: dict[float, list[str]] = field(default_factory=dict)
    feedback_events: int = 0

    def mrr(self, weight: float, kind: str) -> float:
        values = self.ranks[weight][kind]
        return sum(values) / len(values) if values else 0.0


def load_subjects(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise InvalidInput("a feedback replay file has schema_version 1")
    for subject in data["subjects"]:
        if subject["gold"] not in data["passages"]:
            raise InvalidInput(f"subject {subject['id']}: its answering passage is not in the passages")
        if subject["half"] not in HALVES or len(subject["asked"]) < 2:
            raise InvalidInput(f"subject {subject['id']}: a half of a or b and two ways of asking it")
    return data


async def replay(path: Path, *, judged_half: str, judge: str = "strict", judgements: int = 2,
                 weights: tuple[float, ...] = WEIGHTS) -> Replay:
    """One replay: judge ``judged_half`` with ``judge`` from its first ``judgements`` ways of asking, then rank."""
    if judged_half not in HALVES or judge not in JUDGES or judgements not in (1, 2):
        raise InvalidInput("judged_half is a or b, judge is kind or strict, judgements is 1 or 2")
    data = load_subjects(path)
    clock = Clock(START)
    events = InMemoryEventLog()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock,
                                events=events).open()
    try:
        episodes = {text: (await engine.remember(SPACE, text)).episode_id for text in data["passages"]}
        judged = [subject for subject in data["subjects"] if subject["half"] == judged_half]
        for day in range(judgements):
            clock.now = f"2026-05-{2 + day:02d}T09:00:00.000Z"
            for subject in judged:
                shown = await engine.recall(SPACE, subject["asked"][day], limit=SHOWN)
                gold = episodes[subject["gold"]]
                positions = [item.episode_id for item in shown.items]
                above = positions.index(gold) if gold in positions else len(positions)
                assert shown.event_id is not None
                if gold in positions:
                    await engine.feedback(SPACE, shown.event_id, shown.items[above].chunk_id, True)
                if judge == "strict":
                    for item in shown.items[:above]:
                        await engine.feedback(SPACE, shown.event_id, item.chunk_id, False)
        clock.now = f"2026-05-{2 + judgements:02d}T09:00:00.000Z"
        questions = [("judged", subject["id"], subject["paraphrase"], subject["gold"]) for subject in judged]
        questions += [("unrelated", subject["id"], question, subject["gold"])
                      for subject in data["subjects"] if subject["half"] != judged_half
                      for question in (*subject["asked"], subject["paraphrase"])]
        result = Replay(judged_half, judge, judgements,
                        feedback_events=len(await events.query(SPACE, kind="feedback", limit=100_000)))
        every = (0.0, *weights)
        for weight in every:
            result.ranks[weight] = {"judged": [], "unrelated": []}
            result.rose[weight], result.fell[weight] = [], []
        for kind, subject_id, question, gold_text in questions:
            baseline = 0.0
            for weight in every:  # interleaved per question, on one engine and one record
                engine.feedback_weight = weight
                found = await engine.recall(SPACE, question, limit=DEPTH)
                rank = reciprocal_rank([str(item.episode_id) for item in found.items], {str(episodes[gold_text])})
                result.ranks[weight][kind].append(rank)
                if weight == 0.0:
                    baseline = rank
                elif rank != baseline:
                    (result.rose if rank > baseline else result.fell)[weight].append(f"{kind}:{subject_id}:{question}")
        return result
    finally:
        await engine.close()


async def measure(path: Path, *, judge: str = "strict", judgements: int = 2,
                  weights: tuple[float, ...] = WEIGHTS) -> list[Replay]:
    """The replay once per half judged."""
    return [await replay(path, judged_half=half, judge=judge, judgements=judgements, weights=weights) for half in HALVES]


def report(replays: list[Replay]) -> str:
    first = replays[0]
    weights = list(first.ranks)
    lines = [f"judge: {first.judge}; judgements per subject: {first.judgements}",
             "half judged  set        n   " + "  ".join(f"w={weight:<6g}" for weight in weights)]
    for run in replays:
        for kind in ("judged", "unrelated"):
            row = "  ".join(f"{run.mrr(weight, kind):.4f}  " for weight in weights)
            lines.append(f"{run.judged_half:<12} {kind:<10} {len(run.ranks[0.0][kind]):>2}   {row}")
    for kind in ("judged", "unrelated"):
        pooled = "  ".join(f"{sum(run.mrr(weight, kind) for run in replays) / len(replays):.4f}  " for weight in weights)
        lines.append(f"{'both':<12} {kind:<10} {sum(len(run.ranks[0.0][kind]) for run in replays):>2}   {pooled}")
    for weight in weights[1:]:
        rose = sum(len(run.rose[weight]) for run in replays)
        fell = [entry for run in replays for entry in run.fell[weight]]
        lines.append(f"w={weight:g}: {rose} question(s) rose, {len(fell)} fell: {'; '.join(fell) or '-'}")
    return "\n".join(lines)
