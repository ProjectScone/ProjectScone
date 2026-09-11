"""A claim stated again, kept as evidence of when it was stated.

When a claim is asserted from a day on which it already holds, the ledger
returns the fact that holds and writes nothing a reader sees: that is a
restatement. But the later day is evidence too. If a backfill arrives
afterwards and cuts the fact short before that day, the claim must resume
from it, or the order facts arrived in would decide what held (G04). An
affirmation keeps that day, and the evidence behind it, beside the fact.

Affirmations are an optional store capability, like the ledger pager: a
store without them keeps today's behaviour, and a restatement there is
simply returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, cast, runtime_checkable

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class NewAffirmation:
    space: str
    fact_id: int
    valid_from: str
    recorded_at: str
    confidence: float = 1.0
    source_episode_id: Optional[int] = None
    origin: str = "stated"
    quote: Optional[str] = None
    #: The restatement's own dependency links, as (kind, fact id): what it
    #: extends and was derived from. Should the claim resume from this
    #: day, the fact it resumes as rests on these, not on the first fact's.
    links: tuple[tuple[str, int], ...] = ()


class Affirmation(BaseModel):
    """``fact_id``'s claim, stated again as holding from ``valid_from``."""

    affirmation_id: int
    space: str
    fact_id: int
    valid_from: str
    recorded_at: str
    confidence: float = 1.0
    source_episode_id: Optional[int] = None
    origin: str = "stated"
    quote: Optional[str] = None
    links: list[tuple[str, int]] = Field(default_factory=list)


def stored_links(links: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
    """Links as a document store keeps them: named fields, one per link."""
    return [{"kind": kind, "to_fact": to_fact} for kind, to_fact in links]


def read_links(stored: Sequence[Mapping[str, object]]) -> list[tuple[str, int]]:
    """Links back from ``stored_links``."""
    return [(str(link["kind"]), int(cast(int, link["to_fact"]))) for link in stored]


@runtime_checkable
class AffirmationStore(Protocol):
    async def add_affirmation(self, new: NewAffirmation) -> Affirmation:
        """Keep one; the same (space, fact, valid_from) kept again returns
        the first, so stating a claim twice on one day records it once."""
        ...

    async def affirmations(self, space: str, fact_id: int) -> list[Affirmation]:
        """The fact's affirmations, earliest valid_from first."""
        ...

    async def drop_affirmations(self, space: str, affirmation_ids: Sequence[int]) -> None: ...

    async def space_affirmations(self, space: str) -> list[Affirmation]:
        """Every affirmation of the space, for an archive, by id."""
        ...


_METHODS = ("add_affirmation", "affirmations", "drop_affirmations", "space_affirmations")


def affirmation_store(documents: object) -> AffirmationStore | None:
    """The store as an affirmation store, when it is one."""
    return cast(AffirmationStore, documents) if all(callable(getattr(documents, name, None)) for name in _METHODS) \
        else None
