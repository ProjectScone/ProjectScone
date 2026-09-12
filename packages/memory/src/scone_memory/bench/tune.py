"""Retrieval settings chosen by measurement, not by taste.

Every knob this framework has is a choice somebody could argue about:
how many candidates a lane fetches, whether a claim restated many times
is demoted, whether a chunk is embedded with the date and source in
front of it. None of them is right everywhere, and the honest way to
pick one is to measure it on the corpus it will serve.

``tune`` runs the ordinary bench once per setting, over the same
questions and the same sample, and reports what each found. The rule for
choosing is stated rather than implied: the setting that answered most
wins; a tie goes to the quicker one; and a change that only matches the
default is no change at all, so the default stands. What comes out names
the environment lines that put the winner in force — it is a
recommendation with its measurement attached, not a file the engine
reads behind anyone's back.

Nothing here is paid: it runs whatever embedder is configured, over a
dataset already on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

from ..core.errors import InvalidInput

#: Settings a sweep may vary. Anything else stays as configured.
TUNED = ("candidate_limit", "demote_restated", "contextual_embeddings")
#: Questions a sweep may run, per setting.
MAX_SAMPLE = 2_000
#: Settings one sweep may measure. Each is a whole bench run.
MAX_SETTINGS = 24


@dataclass(frozen=True)
class Setting:
    """One combination of the options a sweep varies, against the
    defaults this framework ships with."""

    candidate_limit: Optional[int] = None
    demote_restated: bool = True
    contextual_embeddings: bool = False

    def record(self) -> dict[str, object]:
        return {"candidate_limit": self.candidate_limit, "demote_restated": self.demote_restated,
                "contextual_embeddings": self.contextual_embeddings}

    def environment(self) -> list[str]:
        """The lines that put this setting in force, and nothing for a
        setting that is already the default."""
        lines = []
        if self.candidate_limit is not None:
            lines.append(f"SCONE_RECALL_CANDIDATES={self.candidate_limit}")
        if not self.demote_restated:
            lines.append("SCONE_DEMOTE_RESTATED=0")
        if self.contextual_embeddings:
            lines.append("SCONE_CONTEXTUAL_EMBEDDINGS=1")
        return lines

    def text(self) -> str:
        return ", ".join(self.environment()) or "the defaults"


DEFAULT_SETTINGS = Setting()


@dataclass(frozen=True)
class Measured:
    """What one setting found, on the questions every setting was asked."""

    setting: Setting
    questions: int
    recall_any: float
    recall_all: float
    recall_ms_p50: Optional[float]
    errors: int

    def record(self) -> dict[str, object]:
        return {"setting": self.setting.record(), "questions": self.questions,
                "recall_any": self.recall_any, "recall_all": self.recall_all,
                "recall_ms_p50": self.recall_ms_p50, "errors": self.errors}


@dataclass(frozen=True)
class Tuning:
    """A sweep: what each setting found, which was taken, and why."""

    dataset: str
    embedder: str
    k: int
    sample: int
    #: The sample's seed, so the same questions can be drawn again.
    seed: int = 42
    rows: tuple[Measured, ...] = ()
    chosen: Optional[Setting] = None
    reason: str = ""
    measured: dict[str, object] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        return {"dataset": self.dataset, "embedder": self.embedder, "k": self.k, "sample": self.sample,
                "seed": self.seed,
                "rows": [row.record() for row in self.rows],
                "chosen": self.chosen.record() if self.chosen else None,
                "environment": self.chosen.environment() if self.chosen else [],
                "reason": self.reason, "measured": dict(self.measured)}

    def text(self) -> str:
        """What a person needs: every setting's score, then the one to take
        and the lines that put it in force."""
        lines = [f"tuning: {self.dataset}, {self.sample} question(s) at k={self.k} (seed {self.seed}), "
                 f"embedded by {self.embedder}"]
        for row in sorted(self.rows, key=lambda item: (-item.recall_any, item.recall_ms_p50 or 0.0)):
            lines.append(f"  {row.setting.text()}: recall_any {row.recall_any:.3f}, "
                         f"recall_all {row.recall_all:.3f}"
                         + (f", p50 {row.recall_ms_p50:.1f} ms" if row.recall_ms_p50 is not None else "")
                         + (f", {row.errors} error(s)" if row.errors else ""))
        lines.append(f"take: {self.chosen.text() if self.chosen else 'nothing'} — {self.reason}")
        lines += [f"  {line}" for line in (self.chosen.environment() if self.chosen else [])]
        return "\n".join(lines)


def chosen_setting(rows: Sequence[Measured]) -> tuple[Optional[Setting], str]:
    """The setting to take, and why in words. Most answers wins; a tie goes
    to the quicker; and a change that only matches the default is no change,
    because a setting is a thing to explain and this one would explain
    nothing."""
    if not rows:
        return None, "nothing was measured"
    best = max(row.recall_any for row in rows)
    tied = [row for row in rows if row.recall_any == best]
    asked = max(row.questions for row in rows)
    default = next((row for row in tied if row.setting == DEFAULT_SETTINGS), None)
    if default is not None:
        lowest = min(row.recall_any for row in rows)
        return DEFAULT_SETTINGS, (f"nothing measured better than the defaults ({best:.3f} against "
                                  f"{lowest:.3f} at worst, over {asked} question(s))")
    # A row nobody timed is not a quick row: it goes last among equals.
    quickest = min(tied, key=lambda row: (row.recall_ms_p50 if row.recall_ms_p50 is not None else float("inf"),
                                          row.setting.text()))
    was = next((row.recall_any for row in rows if row.setting == DEFAULT_SETTINGS), None)
    # A rate hides how few questions are behind it: 0.900 against 0.833
    # over thirty questions is two questions, and two questions is how a
    # default gets changed on noise.
    apart = round((quickest.recall_any - was) * quickest.questions) if was is not None else 0
    why = (f"{quickest.recall_any:.3f} against {was:.3f} by the defaults, which is {apart} more "
           f"question(s) of {quickest.questions}" if was is not None
           else f"{quickest.recall_any:.3f}, the most any setting found over {quickest.questions} question(s)")
    if len(tied) > 1:
        why += f"; the quicker of {len(tied)} settings that tied"
    return quickest.setting, why


async def tune(dataset: str | Path, *, settings: Sequence[Setting], k: int = 5, sample: int = 30,
               seed: int = 42, base: object = None, progress: object = None) -> Tuning:
    """Measure every setting on the same questions, and say which to take.

    ``base`` is the configuration everything else is held at; a sweep only
    ever changes the three settings above, so what it measures is the
    difference between them and nothing else."""
    from ..runtime.config import Settings, build_embedder, build_in_process_engine
    from .runner import load_items, stratified_sample, run

    if not 1 <= k <= 100:
        raise InvalidInput("k must be from 1 to 100")
    if not 1 <= sample <= MAX_SAMPLE:
        raise InvalidInput(f"sample must be from 1 to {MAX_SAMPLE} questions")
    if not settings:
        raise InvalidInput("a sweep needs at least one setting to measure")
    if len(settings) > MAX_SETTINGS:
        raise InvalidInput(f"a sweep measures at most {MAX_SETTINGS} settings; each is a whole bench run")
    held = base if isinstance(base, Settings) else Settings.from_env({})
    items = stratified_sample(load_items(dataset), sample, seed=seed)
    embedder = build_embedder(held)
    rows: list[Measured] = []
    for setting in settings:
        under = replace(held, **{name: getattr(setting, name) for name in TUNED})

        async def engine_for(under: "Settings" = under):
            return await build_in_process_engine(under, embedder)

        report = await run(engine_for, items, ks=(k,), dataset=str(dataset))
        rows.append(Measured(setting=setting, questions=report.scored,
                             recall_any=report.recall_any[k], recall_all=report.recall_all[k],
                             recall_ms_p50=report.recall_ms_p50, errors=report.errors))
    chosen, reason = chosen_setting(rows)
    return Tuning(dataset=str(dataset), embedder=embedder.id, k=k, sample=len(items), seed=seed,
                  rows=tuple(rows), chosen=chosen, reason=reason,
                  measured={"questions": rows[0].questions if rows else 0, "settings": len(rows)})
