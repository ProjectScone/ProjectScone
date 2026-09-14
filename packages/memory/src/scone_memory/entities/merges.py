"""Which names a person decided are one entity, read from the ledger.

The identity rule joins two names only when their keys are equal. Deciding
that "Dr. Alice Chen" and "alice chen" are one person is a judgement with
evidence behind it, so it is recorded rather than guessed: a claim that the
alias is the same entity as the name it is merged into, under a predicate
no ordinary assertion may use. Being a claim, it has a valid time, a status
and a history; closing it parts the names again from that moment, and a
view of an earlier moment still sees them joined while the decision held.

This module only reads decisions. Every view that projects a ledger applies
the same reading, so the graph a person sees and the one retrieval walks
agree on which names are one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Mapping

from ..core.models import Fact
from ..core.timeutil import parse_rfc3339
from ..core.validation import entity_key

__all__ = ["SAME_ENTITY", "EntityMerges", "MergeDecision", "entity_merges", "is_decision"]

#: The predicate a merge is recorded under, as the ledger stores predicates.
#: The prefix keeps it apart from anything a person or a model would state:
#: a document saying "X same as Y" is a claim to weigh, not a decision.
SAME_ENTITY = "scone:same entity"

MergeOutcome = Literal["applied", "cycle", "same_name"]


@dataclass(frozen=True, slots=True)
class MergeDecision:
    fact_id: int
    alias_key: str
    #: The key the alias was merged into, as recorded.
    named_key: str
    #: Where the alias ends up once every chain is followed.
    into_key: str
    outcome: MergeOutcome


@dataclass(frozen=True)
class EntityMerges:
    #: Alias key -> the key it resolves to, chains followed.
    into: Mapping[str, str]
    decisions: tuple[MergeDecision, ...]

    def canonical(self, key: str) -> str:
        return self.into.get(key, key)

    def keys(self) -> frozenset[str]:
        """Every key an applied decision names, aliases and targets alike."""
        return frozenset(self.into) | frozenset(self.into.values())

    def group(self, key: str) -> tuple[str, ...]:
        """The key's entity and every alias merged into it, the entity first."""
        target = self.canonical(key)
        return (target, *sorted(alias for alias, into in self.into.items() if into == target))

    def __bool__(self) -> bool:
        return bool(self.decisions)


NO_MERGES = EntityMerges({}, ())


def is_decision(fact: Fact) -> bool:
    return fact.predicate == SAME_ENTITY


def _holds(fact: Fact, at: datetime | None) -> bool:
    if fact.excluded or fact.status not in ("active", "closed"):
        return False
    if at is None:
        return fact.status == "active" and fact.valid_until is None
    return parse_rfc3339(fact.valid_from) <= at and (fact.valid_until is None or parse_rfc3339(fact.valid_until) > at)


def entity_merges(facts: Iterable[Fact], at: datetime | None = None) -> EntityMerges:
    """The merges in force at ``at`` (with None, those in force now: confirmed
    and still open), in the order they were recorded. A decision whose target
    already resolves back to its alias would close a cycle and is refused,
    reported rather than dropped."""
    pointer: dict[str, str] = {}
    recorded: list[tuple[int, str, str, MergeOutcome]] = []

    def chain(key: str) -> list[str]:
        walked = [key]
        # A loop is refused before it is recorded, so a repeat never comes;
        # stopping at one keeps a broken map from hanging every view.
        while key in pointer and pointer[key] not in walked:
            key = pointer[key]
            walked.append(key)
        return walked

    def resolve(key: str) -> str:
        return chain(key)[-1]

    for fact in sorted((fact for fact in facts if is_decision(fact) and _holds(fact, at)),
                       key=lambda fact: fact.fact_id):
        alias, named = entity_key(fact.subject), entity_key(fact.object)
        if alias == named:
            recorded.append((fact.fact_id, alias, named, "same_name"))
        elif alias in chain(named):
            # The alias's own pointer is replaced, never followed, so this is
            # the only way a new pointer can loop.
            recorded.append((fact.fact_id, alias, named, "cycle"))
        else:
            pointer[alias] = named
            recorded.append((fact.fact_id, alias, named, "applied"))
    into = {alias: resolve(alias) for alias in pointer}
    return EntityMerges(into, tuple(
        MergeDecision(fact_id, alias, named, into.get(alias, alias) if outcome == "applied" else named, outcome)
        for fact_id, alias, named, outcome in recorded))
