"""What a predicate means, beyond the claims that use it.

A ledger records what was said. "Alice Chen works at Acme Robotics" says
nothing, on its own, about what Acme employs, whether Bob is married to
Alice when Alice is married to Bob, or where a shelf is when its aisle is
in a warehouse. People know those things because they know what the words
mean, and a graph that does not is a record of claims rather than
something to ask questions of.

Three meanings are enough for most of it, and each is configured, never
guessed: this never decides that two predicates are opposites because
they look alike.

* **opposite** (inverse) — `works_at` is the other side of `employs`.
* **reads both ways** (symmetric) — `married_to` holds either way round.
* **carries through** (transitive) — a shelf in an aisle in a warehouse
  is in the warehouse.

What follows from a claim is never written into the ledger and never
mixed with it: implied relations are projected beside the stated ones,
each naming the claims it rests on and the relations it follows from, so
anything reading the graph can tell what was said from what was worked
out. Every implied relation is therefore as good as its evidence and can
be traced to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from ..core.errors import InvalidInput
from ..core.validation import normalise_term

#: How far a meaning that carries through is followed. Four steps covers
#: the containments people write down (shelf, aisle, warehouse, estate)
#: and keeps the work a small multiple of the ledger rather than its
#: square.
MAX_STEPS = 4
#: Predicates a vocabulary may name, all told.
MAX_MEANINGS = 500
#: Relations that may be implied for one projection. Past this the graph
#: says what it left out rather than growing without bound.
MAX_IMPLIED = 50_000
#: Claims a whole walk may examine. A dense graph has more paths than
#: anyone can walk, and finding nothing new is not the same as being
#: cheap, so the work itself is bounded and what it stopped short of is
#: said rather than assumed.
MAX_WALKED = 200_000


def _named(predicates: object, what: str) -> list[str]:
    if isinstance(predicates, (str, bytes)):
        raise InvalidInput(f"{what} takes a collection of predicates, not one string")
    if not isinstance(predicates, Iterable):
        raise InvalidInput(f"{what} takes a collection of predicates")
    return [normalise_term(str(name), f"a predicate in {what}") for name in predicates]


@dataclass(frozen=True)
class RelationMeanings:
    """A space's vocabulary: which predicates are opposites, which read the
    same both ways, and which carry through. Checked where it is built, so
    a vocabulary that cannot mean what it says is refused before a graph is
    read by it."""

    inverse: Mapping[str, str] = field(default_factory=dict)
    symmetric: Iterable[str] = ()
    transitive: Iterable[str] = ()

    def __post_init__(self) -> None:
        both = frozenset(_named(self.symmetric, "symmetric"))
        through = frozenset(_named(self.transitive, "transitive"))
        if not isinstance(self.inverse, Mapping):
            raise InvalidInput("inverse takes a mapping of predicate to its opposite")
        opposites: dict[str, str] = {}
        for left, right in self.inverse.items():
            one = normalise_term(str(left), "a predicate in inverse")
            other = normalise_term(str(right), "the opposite of a predicate")
            if one == other:
                raise InvalidInput(
                    f"{one!r} cannot be its own opposite; a predicate that holds either way round "
                    f"reads the same both ways, so name it in symmetric")
            for side, partner in ((one, other), (other, one)):
                if side in both:
                    raise InvalidInput(f"{side!r} reads the same both ways, so it has no other side to name")
                if opposites.get(side, partner) != partner:
                    raise InvalidInput(f"{side!r} may have one opposite, not {opposites[side]!r} and {partner!r}")
                opposites[side] = partner
        if len(opposites) + len(both) + len(through) > MAX_MEANINGS:
            raise InvalidInput(f"a vocabulary may name at most {MAX_MEANINGS} predicates")
        # A read-only mapping, not a dict. The class is frozen, but a plain
        # dict behind a frozen field is not immutable: a holder could empty
        # it and leave a cached projection implying edges the vocabulary no
        # longer mentions. Nothing in the framework mutates this mapping,
        # so making it refuse is free.
        object.__setattr__(self, "inverse", MappingProxyType(dict(sorted(opposites.items()))))
        object.__setattr__(self, "symmetric", tuple(sorted(both)))
        object.__setattr__(self, "transitive", tuple(sorted(through)))

    def __bool__(self) -> bool:
        return bool(self.inverse or self.symmetric or self.transitive)

    def opposite(self, predicate: str) -> str | None:
        """The other side of a predicate: the one named as its opposite, or
        itself when it reads the same both ways."""
        if predicate in self.symmetric:
            return predicate
        got = self.inverse.get(predicate)
        return got

    def reads_both_ways(self, predicate: str) -> bool:
        return predicate in self.symmetric

    def carries_through(self, predicate: str) -> bool:
        return predicate in self.transitive

    def record(self) -> dict[str, object]:
        """The vocabulary as it will be applied, which is what a projection
        is digested with and what a reader is shown."""
        return {"inverse": dict(self.inverse), "symmetric": list(self.symmetric),
                "transitive": list(self.transitive), "max_steps": MAX_STEPS, "max_implied": MAX_IMPLIED,
                "max_walked": MAX_WALKED}
