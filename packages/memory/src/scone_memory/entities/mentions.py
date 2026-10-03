"""The predicates a named-entity recognizer records its mentions under.

A record that names Acme says nothing about Acme; it says the record
names it. So a mention is written as a claim about the record -- its
source, or the episode when it has none -- under a predicate of its own:
``scone:mentions organisation`` and so on, one per kind a recognizer can
give, and a bare ``scone:mentions`` when it named something no kind here
fits. The kind rides on the predicate because a fact has nowhere else to
carry it, and that is where every other kind hint already comes from
(``kinds.hint``): the projection reads a recognized kind as one more hint,
with the same conflict rule as any other.

The prefix keeps these apart from anything a person or a model states,
as with merge decisions: ordinary assertion refuses them, so a kind read
from a ``scone:mentions`` fact was always the recognizer's. And a mention
is an index entry rather than a claim about the world, so fact recall, the
profile, derivation and the distiller's queue leave it out; only the
entity graph reads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.models import Fact
    from .kinds import EntityKind

__all__ = ["MENTIONS", "RECOGNIZED_KINDS", "is_mention", "is_mention_predicate", "mention_kind", "mention_predicate"]

#: The predicate of a mention whose kind is unknown, as the ledger stores
#: predicates; a kind follows it after one space.
MENTIONS = "scone:mentions"
#: The kinds a recognizer's mention can carry on its predicate.
RECOGNIZED_KINDS: tuple["EntityKind", ...] = ("person", "organisation", "place", "product", "event", "nationality")
_BY_PREDICATE: dict[str, "EntityKind"] = {f"{MENTIONS}_{kind}": kind for kind in RECOGNIZED_KINDS}


def mention_predicate(kind: Optional["EntityKind"]) -> str:
    """The predicate a mention of this kind is recorded under."""
    if kind is None:
        return MENTIONS
    if kind not in RECOGNIZED_KINDS:
        raise ValueError(f"{kind!r} is not a kind a recognizer records")
    return f"{MENTIONS} {kind}"


def mention_kind(predicate: str) -> Optional["EntityKind"]:
    """The kind a mention predicate carries, spelled as stored or with
    underscores (``kinds.hint`` reads it that way), or None."""
    return _BY_PREDICATE.get(predicate.replace(" ", "_"))


def is_mention_predicate(predicate: str) -> bool:
    folded = predicate.replace(" ", "_")
    return folded == MENTIONS or folded in _BY_PREDICATE


def is_mention(fact: "Fact") -> bool:
    return is_mention_predicate(fact.predicate)
