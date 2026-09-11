"""Measuring the floor an engine abstains by, rather than guessing it.

Recall is run over questions whose answer is in memory and, for each,
one question whose answer is not, so both sides of the judgement are
measured on the same corpus with the same embedder. Each candidate floor
is scored by how many unanswerable questions it would catch and how many
answerable ones it would withhold, and the floor taken is the highest
one inside the budget for withheld answers.

The policy that comes out names the embedder it was measured with, so an
engine that embeds differently refuses it instead of reading another
embedder's scale as its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence, cast

from ..retrieval.abstention import AbstentionPolicy, PolicyError, choose_floor
from .runner import BenchItem, RunReport, run

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: The share of answerable questions a floor may withhold, by default.
DEFAULT_TARGET = 0.05


async def calibrate(make_engine: Callable[[], object], items: Sequence[BenchItem], *,
                    target_false_abstain: float = DEFAULT_TARGET, dataset: str = "",
                    ks: Sequence[int] = (5,)) -> tuple[Optional[AbstentionPolicy], RunReport]:
    """A policy measured on these questions, or None when no floor is
    inside the budget; the report it was measured from comes back too."""
    import inspect

    probe = make_engine()
    engine = cast("MemoryEngine", await probe if inspect.isawaitable(probe) else probe)
    embedder_id, dim = engine.embedder.id, engine.embedder.dim
    await engine.close()
    report = await run(make_engine, items, ks=tuple(ks), dataset=dataset, cross_queries=True)  # type: ignore[arg-type]
    if report.embedder and report.embedder != embedder_id:
        raise PolicyError(f"the run embedded with {report.embedder}, not the {embedder_id} this measured")
    sweep = report.abstention
    if not sweep:
        return None, report
    floor = choose_floor(sweep, target_false_abstain=target_false_abstain)
    if floor is None:
        return None, report
    return AbstentionPolicy(
        floor=floor, embedder_id=embedder_id, dim=dim,
        measured={"questions": report.items, "answerable": sweep.get("evidence_n"),
                  "unanswerable": sweep.get("no_evidence_n"),
                  "abstain_rate": _rate(sweep, "abstain_rate", floor),
                  "false_abstain_rate": _rate(sweep, "false_abstain_rate", floor),
                  "target_false_abstain": target_false_abstain, "dataset": dataset or report.dataset,
                  "measured_at": report.finished_at}), report


def _rate(sweep: dict, name: str, floor: float) -> Optional[float]:
    rates = sweep.get(name) or {}
    for key, value in rates.items():
        if float(key) == floor:
            return None if value is None else float(value)
    return None


def write_policy(policy: AbstentionPolicy, path: str | Path) -> Path:
    """Write a policy where an engine can be pointed at it."""
    import json

    written = Path(path)
    written.write_text(json.dumps(policy.record(), indent=2) + "\n", encoding="utf-8")
    return written
