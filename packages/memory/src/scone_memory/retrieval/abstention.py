"""A measured floor for abstaining, tied to the embedder that measured it.

A similarity is a number one embedder produces under one set of settings.
It is not a probability, and no floor is right for every embedder or
every corpus, so this framework guesses none. A floor is chosen by
measurement: recall is run over questions whose answer is in memory and
questions whose answer is not, each candidate floor is scored by how many
of the second it would catch and how many of the first it would withhold,
and the floor taken is the highest one still inside the budget for
withheld answers.

What comes out is a record, not a number: the floor, the embedder it was
measured with, and what it cost on how many questions. An engine that
embeds differently refuses it rather than reading another embedder's
scale as its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Optional, cast

SCHEMA_VERSION = 1
#: A policy is a few hundred bytes; a larger file is not one.
MAX_POLICY_BYTES = 64 * 1024


class PolicyError(ValueError):
    """A policy that cannot be trusted, and says why."""


def choose_floor(sweep: Mapping[str, object], *, target_false_abstain: float) -> Optional[float]:
    """The highest floor whose share of wrongly withheld answers is within
    the budget, or None when even the lowest costs more than that."""
    if not 0.0 <= target_false_abstain <= 1.0:
        raise PolicyError("target_false_abstain must be from 0 to 1")
    withheld = sweep.get("false_abstain_rate")
    if not isinstance(withheld, Mapping) or not withheld:
        raise PolicyError("the sweep has no measured false_abstain_rate to choose by")
    # The rates in a sweep are rounded for reading, so the counts decide
    # when they are there: one answer withheld of 21 is 0.047619…, and a
    # target of 0.0476 must not take a floor that cost more than it.
    counted, answerable = sweep.get("false_abstain_n"), sweep.get("evidence_n")
    exact: Optional[Mapping[object, object]] = (
        counted if isinstance(counted, Mapping) and isinstance(answerable, int) and answerable > 0 else None)
    answered = answerable if isinstance(answerable, int) else 0

    def costs(floor: object, rate: object) -> float:
        """What that floor withheld: the count when the sweep carries it."""
        if exact is not None and floor in exact:
            return float(cast(float, exact[floor])) / answered
        return float(cast(float, rate))

    affordable = [float(cast(float, floor)) for floor, rate in withheld.items()
                  if rate is not None and costs(floor, rate) <= target_false_abstain]
    return max(affordable) if affordable else None


@dataclass(frozen=True)
class AbstentionPolicy:
    """A floor, the embedder it was measured with, and what it cost.

    Checked wherever it is built, not only when read from a file: an
    engine takes a policy's floor over its own, so a floor that is not a
    similarity would pass every check the engine makes on its own."""

    floor: float
    embedder_id: str
    dim: int
    measured: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (isinstance(self.floor, bool) or not isinstance(self.floor, (int, float))
                or not math.isfinite(self.floor) or not -1.0 <= float(self.floor) <= 1.0):
            raise PolicyError(f"floor must be a cosine similarity in [-1, 1], got {self.floor!r}")
        if not isinstance(self.embedder_id, str) or not self.embedder_id.strip():
            raise PolicyError("a policy must name the embedder it was measured with")
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 1:
            raise PolicyError(f"dim must be the embedder's width, a positive whole number, got {self.dim!r}")

    def record(self) -> dict[str, object]:
        return {"schema_version": SCHEMA_VERSION, "floor": self.floor, "embedder_id": self.embedder_id,
                "dim": self.dim, "measured": dict(self.measured)}

    def text(self) -> str:
        """One line for an operator: what the floor is and what it cost."""
        measured = self.measured
        cost = measured.get("false_abstain_rate")
        caught = measured.get("abstain_rate")
        return (f"abstention: floor {self.floor} for embedder {self.embedder_id} ({self.dim}-d), "
                f"measured on {measured.get('questions', 'unknown')} questions: "
                f"catches {caught if caught is not None else 'unmeasured'} of the unanswerable, "
                f"withholds {cost if cost is not None else 'unmeasured'} of the answerable")

    def fits(self, embedder_id: str, dim: int) -> bool:
        """Whether this policy may be used by that embedder. A width of 0
        is one the embedder has not learned yet (a remote one learns it
        from its first answer), and is not a mismatch here. It is not a
        way in either: the engine asks again before every recall, and the
        width of the query's own vector is checked against the policy's,
        so a floor is never applied to another width's similarities."""
        return self.embedder_id == embedder_id and dim in (0, self.dim)

    @classmethod
    def read(cls, path: str | Path) -> "AbstentionPolicy":
        """A policy from a file, refused rather than guessed at when its
        version, floor, embedder or width is not one this can use."""
        found = Path(path)
        try:
            size = found.stat().st_size
        except OSError as unreadable:
            raise PolicyError(f"the policy could not be read: {unreadable}") from None
        if size > MAX_POLICY_BYTES:
            raise PolicyError(f"the policy file is too large to be one: {size} bytes, over {MAX_POLICY_BYTES}")
        try:
            written = json.loads(found.read_text(encoding="utf-8"))
        except (OSError, ValueError) as unreadable:
            raise PolicyError(f"the policy could not be read: {unreadable}") from None
        if not isinstance(written, dict):
            raise PolicyError("a policy must be a JSON object")
        if written.get("schema_version") != SCHEMA_VERSION:
            raise PolicyError(f"schema_version must be {SCHEMA_VERSION}, got {written.get('schema_version')!r}")
        floor = written.get("floor")
        if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not -1.0 <= float(floor) <= 1.0:
            raise PolicyError("floor must be a cosine similarity in [-1, 1]")
        embedder_id = written.get("embedder_id")
        if not isinstance(embedder_id, str) or not embedder_id.strip():
            raise PolicyError("the policy must name the embedder it was measured with")
        dim = written.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise PolicyError("dim must be the embedder's width, a positive whole number")
        measured = written.get("measured")
        try:
            return cls(float(floor), embedder_id, dim, dict(measured) if isinstance(measured, dict) else {})
        except PolicyError as refused:  # a floor of NaN reads as a float and is not one
            raise PolicyError(f"the policy could not be used: {refused}") from None
