"""Which facts touch an entity, as its subject or as its object.

The ledger is indexed by subject, so traversal can follow a fact's object to
the facts about that thing, but not ask which facts point at a thing. A role
index answers both, newest first, from a ledger read the projection cache
already holds. It is a candidate generator only: whoever uses it re-reads
each fact and applies the join rule itself, so an index that lags the
ledger can miss a fact but never make one up.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal, Mapping

from ..core.validation import entity_key
from ..memory.identity import lookup_keys
from .read import LedgerRead


@dataclass(frozen=True)
class RoleIndex:
    space: str
    #: The revision of the ledger read it was built from.
    revision: int
    subjects: Mapping[str, tuple[int, ...]]
    objects: Mapping[str, tuple[int, ...]]

    def facts_touching(self, name: str, *, role: Literal["subject", "object"], limit: int) -> tuple[int, ...]:
        """Fact ids, newest first, whose subject (or joinable object) has
        the key of ``name``."""
        table = self.subjects if role == "subject" else self.objects
        return table.get(entity_key(name), ())[:max(0, limit)]

    def degree(self, name: str) -> int:
        key = entity_key(name)
        return len(self.subjects.get(key, ())) + len(self.objects.get(key, ()))

    def restamped(self, revision: int) -> "RoleIndex":
        return replace(self, revision=revision)


def role_index(ledger: LedgerRead) -> RoleIndex:
    """Every fact under its subject's key, and under its object's key when
    the join rule lets the object name an entity at all."""
    subjects: dict[str, list[int]] = defaultdict(list)
    objects: dict[str, list[int]] = defaultdict(list)
    for fact in sorted(ledger.facts, key=lambda fact: fact.fact_id, reverse=True):
        subjects[entity_key(fact.subject)].append(fact.fact_id)
        if lookup_keys(fact.object):
            objects[entity_key(fact.object)].append(fact.fact_id)
    return RoleIndex(ledger.space, ledger.revision, {key: tuple(ids) for key, ids in subjects.items()},
                     {key: tuple(ids) for key, ids in objects.items()})
