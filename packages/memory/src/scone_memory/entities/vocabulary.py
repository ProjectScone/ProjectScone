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
from types import MappingProxyType
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


def _snapshot(meanings: Optional[RelationMeanings]) -> Optional[RelationMeanings]:
    """A copy nobody can mutate into the cache.

    The held vocabulary used to be handed out by reference, so a caller
    could empty its ``inverse`` and leave a cached projection implying
    edges the reported vocabulary no longer mentioned -- under the same
    identity. ``RelationMeanings`` is frozen but its mapping is not, so
    the mapping is wrapped too.
    """
    if meanings is None:
        return None
    return RelationMeanings(inverse=dict(meanings.inverse),
                            symmetric=tuple(meanings.symmetric),
                            transitive=tuple(meanings.transitive))


def _identity(source: str, meanings: Optional[RelationMeanings],
              at: Optional[str] = None) -> str:
    """A short digest of what is in force, for a cache key. ``at`` carries
    the saved revision, so a view built under one revision is never served
    for another even if the meanings happen to look alike."""
    said = json.dumps({"source": source, "at": at,
                       "meanings": None if meanings is None else meanings.record()},
                      sort_keys=True, default=str)
    return hashlib.sha256(said.encode()).hexdigest()[:16]


async def resolve_vocabulary(engine: "MemoryEngine", space: str) -> "Vocabulary":
    """Read the space's vocabulary from the store and hold it.

    Called when an engine opens, on the thread that owns the store's
    connection. A store of this kind is bound to its creating thread, so
    a request served on another thread must not reach it -- and a cache
    that could go stale without saying so would be worse than one read at
    a known moment.

    **The contract, stated rather than implied: a vocabulary saved after
    an engine opened takes effect when that engine is reopened.** That is
    how configuration usually behaves, it costs nothing per request, and
    it still fixes the thing that was broken -- two processes reading one
    space now agree, instead of each applying whatever it was told.
    """
    held = await _read(engine, space)
    kept = getattr(engine, "_vocabulary_read", None)
    if kept is None:
        kept = {}
        setattr(engine, "_vocabulary_read", kept)
    kept[space] = held
    return held


async def read_vocabulary(engine: "MemoryEngine", space: str) -> Vocabulary:
    """The meanings in force for this space, and where they came from.

    Returns what :func:`resolve_vocabulary` settled for this space if it
    has been resolved, so a request never reaches a thread-bound store.
    """
    check_space(space)
    process = engine.relation_meanings
    resolved = getattr(engine, "_vocabulary_read", None)
    if resolved is not None and space in resolved:
        return resolved[space]
    if getattr(engine, "vocabulary", None) is not None:
        # A store of this kind is bound to the thread that opened it, so a
        # request served elsewhere must not reach one. A space nobody named
        # at open therefore falls back -- and **says** it fell back, rather
        # than serving process configuration as though the space had
        # nothing saved. Silence there would be a false statement about a
        # space that may well hold a vocabulary.
        process = engine.relation_meanings
        return Vocabulary(
            meanings=_snapshot(process), source="process" if process is not None else "none",
            why=("this space was not read when this engine opened, so whatever it holds is not "
                 "applied here; name it in vocabulary_spaces to have it read. "
                 + ("this process's configuration applies meanwhile"
                    if process is not None else
                    "nothing is configured in this process, so the graph holds only what was "
                    "said")),
            identity=_identity("process" if process is not None else "none", process,
                               f"{space}:unresolved"))
    return await _read(engine, space)


async def _read(engine: "MemoryEngine", space: str) -> Vocabulary:
    """Work out what is in force, reading the store if one is attached."""
    process = engine.relation_meanings
    kept = getattr(engine, "vocabulary", None)
    if kept is not None:
        saved = kept.get(space)
        if saved is not None and not saved.cleared:
            held = saved.meanings
            return Vocabulary(
                meanings=_snapshot(held), source="space",
                why=f"the space holds this vocabulary, saved at {saved.saved_at.isoformat()} "
                    f"(revision {saved.revision}); every reader of the space applies it, "
                    f"whatever each process was configured with",
                identity=_identity("space", held, f"{space}:{saved.revision}"))
        if saved is not None:
            return Vocabulary(
                meanings=_snapshot(process), source="process" if process is not None else "none",
                why=(f"the space's vocabulary was cleared at {saved.saved_at.isoformat()}, so "
                     + ("this process's configuration applies" if process is not None
                        else "the graph holds only what was said")),
                identity=_identity("process" if process is not None else "none", process,
                                   f"{space}:cleared:{saved.revision}"))
    if process is None:
        return Vocabulary(
            meanings=None, source="none",
            why="nothing is configured in this process, so the graph holds only what was said",
            identity=_identity("none", None))
    return Vocabulary(
        meanings=_snapshot(process), source="process",
        why="these are this process's configuration; a space cannot yet hold its own, so "
            "another process configured differently will read this space differently",
        identity=_identity("process", process))
