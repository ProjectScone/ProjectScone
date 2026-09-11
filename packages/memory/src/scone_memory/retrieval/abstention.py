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
from pathlib import Path
from typing import Mapping, Optional

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
    affordable = [float(floor) for floor, cost in withheld.items()
                  if cost is not None and float(cost) <= target_false_abstain]
    return max(affordable) if affordable else None


@dataclass(frozen=True)
class AbstentionPolicy:
    """A floor, the embedder it was measured with, and what it cost."""

    floor: float
    embedder_id: str
    dim: int
    measured: dict[str, object] = field(default_factory=dict)

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
        return self.embedder_id == embedder_id and self.dim == dim

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
        return cls(float(floor), embedder_id, dim, dict(measured) if isinstance(measured, dict) else {})
