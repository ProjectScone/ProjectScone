"""Where a space's relation vocabulary comes from, said out loud.

What a predicate means to another predicate decides which edges a reader
is shown. Configure ``works_at`` as the inverse of ``employs`` and the
graph answers "who works here" from claims that never said it — which is
right, and is why the configuration matters as much as the claims do.

It comes from process configuration, which means **two processes reading
one space can hand two readers different graphs**. That is a real
correctness problem and it is not fixed here. What is fixed is that it is
no longer invisible: every projection says which vocabulary it was built
under and where that came from, so two disagreeing readers can discover
that they disagree and why.

Why the vocabulary is not stored in the space yet
-------------------------------------------------
It should be, and the obvious home was the event log: every store that
keeps events has one, and no migration is needed. Four rounds of review
established that this cannot be made honest, and the reason is worth
recording so nobody spends the day I spent:

- An event log is allowed to evict. ``InMemoryEventLog`` is a ring;
  ``SqliteEventLog`` and ``MongoEventLog`` take ``max_age_days``. The one
  record holding a space's configuration is as droppable as any other,
  and a reader would then be told the space holds no vocabulary — which
  by then is false.
- Guarding on the constructor's arguments does not work either, because
  **retention is a property of the store, not of the handle**. A Mongo TTL
  index persists after the handle that created it is gone. Two SQLite
  handles on one file can disagree: the one opened without expiry
  promises to keep, and the one opened with expiry sweeps what the first
  saved. SQLite persists nothing about retention for the first handle to
  inspect, so there is no sound promise available to make.

So a vocabulary saved in a space needs a **durable per-space
configuration store** — somewhere retention is governed explicitly rather
than swept, and whose state a reader can verify rather than infer from
whoever opened it. ``DocumentStore`` has no such facility today; adding
one means the protocol and six backends, four of which cannot be
exercised on this machine. That is the work, and it is not a patch to
this module.

Until then this reports honestly rather than storing dishonestly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Optional

from ..memory.engine import check_space
from .meanings import RelationMeanings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine


@dataclass(frozen=True)
class Vocabulary:
    """The meanings in force for a space, and where they came from."""

    meanings: Optional[RelationMeanings]
    #: ``process`` when they come from this process's configuration,
    #: ``none`` when there are none. ``space`` is reserved for the day a
    #: space can hold its own; nothing returns it yet.
    source: str
    why: str
    #: Changes whenever the vocabulary does, so a view cached under one set
    #: of meanings is never served for another.
    identity: str

    def record(self) -> dict[str, object]:
        return {"source": self.source, "why": self.why, "identity": self.identity,
                # `is None`, never truthiness: RelationMeanings() is falsy,
                # and a reader handed `meanings: null` cannot tell "there
                # are none" from "nothing was said".
                "meanings": None if self.meanings is None else self.meanings.record()}


def _identity(source: str, meanings: Optional[RelationMeanings]) -> str:
    """A short digest of what is in force, for a cache key."""
    said = json.dumps({"source": source,
                       "meanings": None if meanings is None else meanings.record()},
                      sort_keys=True, default=str)
    return hashlib.sha256(said.encode()).hexdigest()[:16]


async def read_vocabulary(engine: "MemoryEngine", space: str) -> Vocabulary:
    """The meanings in force for this space, and where they came from.

    Reads nothing from any store: the answer is this process's
    configuration, which is the honest answer while a space cannot hold
    one of its own.
    """
    check_space(space)
    process = engine.relation_meanings
    if process is None:
        return Vocabulary(
            meanings=None, source="none",
            why="nothing is configured in this process, so the graph holds only what was said",
            identity=_identity("none", None))
    return Vocabulary(
        meanings=process, source="process",
        why="these are this process's configuration; a space cannot yet hold its own, so "
            "another process configured differently will read this space differently",
        identity=_identity("process", process))
