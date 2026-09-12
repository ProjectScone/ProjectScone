"""A space's relation vocabulary, held by the space rather than the caller.

What a predicate means to another predicate decides which edges a reader
is shown. Configure ``works_at`` as the inverse of ``employs`` and the
graph answers "who works here" from claims that never said it — which is
right, and is exactly why the configuration matters as much as the claims
do.

Until now that configuration came from the environment at engine
construction, so the CLI and the server could hold **different**
vocabularies for the **same** space and hand two readers different
graphs. One space with two meanings is not a preference to be configured;
it is a correctness problem, and the space is the only thing the two
readers share.

So a vocabulary can be **saved into the space**, and what the space holds
beats what a process was told. Three things this gets right:

- **Every answer says where its vocabulary came from** — the space, this
  process, or nowhere. Without that a reader cannot tell why two
  processes disagreed, and could not have told that they did.
- **A store that keeps no events refuses the save** rather than accepting
  one that goes nowhere. Reading still works and reports that it fell
  back, because "nothing was saved" and "nothing could be saved" are
  different facts.
- **The vocabulary has an identity that changes when it does.** Saving one
  is not a write to the ledger, so a cache keyed on the space's revision
  would never notice; a cache that does not include this identity makes a
  save a silent no-op, which is worse than not offering it at all.

Nothing here calls a model. The record is an event, which every store
that keeps events already has, rather than a new column somebody has to
migrate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Optional

from ..core.errors import InvalidInput, SconeError
from ..memory.engine import check_space
from .meanings import RelationMeanings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: The event a saved vocabulary is written as.
SAVED = "graph.meanings"
#: Events read back when looking for the latest save. One space's
#: vocabulary changes rarely; this is a ceiling, not an expectation.
MAX_READ = 200
#: What an event log may say it drops. A log that declares either of these
#: will evict eventually -- a ring filling with unrelated traffic in
#: another space, or an age limit passing -- and the record holding this
#: space's configuration is as evictable as any other. A reader would then
#: be told the space holds no vocabulary, which by then would be false.
#:
#: Configuration a reader depends on cannot live somewhere that may forget
#: it, so such a log refuses the save instead of losing it later.
#: ``SqliteEventLog`` and ``MongoEventLog`` are unbounded unless an
#: operator sets ``max_age_days``; ``InMemoryEventLog`` is always a ring
#: and so can never hold one.
_RETENTION = ("max_events", "max_age_days")


def _forgets(log: object) -> Optional[str]:
    """What this log admits it may drop, or None when it promises to keep."""
    for named in _RETENTION:
        held = getattr(log, named, None)
        if isinstance(held, (int, float)) and held > 0:
            return f"keeps at most {held:g} ({named})"
    return None


class NoEventLog(SconeError):
    """Raised when a vocabulary cannot be saved because nothing would keep
    it. Distinct from a space that simply has none saved."""


@dataclass(frozen=True)
class Vocabulary:
    """The meanings in force for a space, and where they came from."""

    meanings: Optional[RelationMeanings]
    #: ``space`` when the space holds them, ``process`` when they come from
    #: this process's configuration, ``none`` when there are none.
    source: str
    why: str
    #: Changes whenever the vocabulary does, so a cache can key on it. Two
    #: processes reading the same saved vocabulary compute the same value.
    identity: str
    #: When the space's vocabulary was last written, if it was.
    at: Optional[str] = None

    def record(self) -> dict[str, object]:
        return {"source": self.source, "why": self.why, "identity": self.identity,
                "at": self.at,
                # `is None`, never truthiness: RelationMeanings() is falsy,
                # and a reader handed `meanings: null` cannot tell "the space
                # says there are none" from "the space says nothing".
                "meanings": None if self.meanings is None else self.meanings.record()}


def _identity(source: str, meanings: Optional[RelationMeanings], at: Optional[str]) -> str:
    """A short digest of what is in force. Includes the source and the save
    time, so falling back to a process configuration and holding the same
    vocabulary in the space are not the same identity."""
    said = json.dumps({"source": source, "at": at,
                       "meanings": None if meanings is None else meanings.record()},
                      sort_keys=True, default=str)
    return hashlib.sha256(said.encode()).hexdigest()[:16]


async def save_meanings(engine: "MemoryEngine", space: str,
                        meanings: RelationMeanings) -> Vocabulary:
    """Write these meanings into the space, for every reader of it."""
    check_space(space)
    if not isinstance(meanings, RelationMeanings):
        raise InvalidInput("a vocabulary is a RelationMeanings")
    return await _write(engine, space, meanings)


async def clear_meanings(engine: "MemoryEngine", space: str) -> Vocabulary:
    """Record that the space holds no vocabulary of its own.

    Written as a record rather than by deleting the old one: a reader
    should be able to tell "cleared on Tuesday" from "never had one", and
    the events are append-only anyway.
    """
    check_space(space)
    return await _write(engine, space, None)


async def _write(engine: "MemoryEngine", space: str,
                 meanings: Optional[RelationMeanings]) -> Vocabulary:
    if engine.events is None:
        raise NoEventLog(
            f"this engine keeps no events, so a vocabulary saved for {space!r} would be kept "
            f"nowhere and every reader would carry on with its own configuration. Attach an "
            f"event log to hold one.")
    forgets = _forgets(engine.events)
    if forgets is not None:
        raise NoEventLog(
            f"this event log {forgets}, so a vocabulary saved for {space!r} could be evicted by "
            f"unrelated traffic and every reader would silently fall back to its own "
            f"configuration. Refusing to keep configuration somewhere that may forget it: use a "
            f"log without a retention bound, or leave the vocabulary to process configuration.")
    # A space that has been deleted takes no writes. A save that reported
    # success into one would be a claim about durable state that nothing
    # will ever honour.
    await engine._living(space)
    # `is None` and not truthiness: `RelationMeanings()` is **falsy**, so
    # testing it for truth turned an explicitly empty vocabulary into a
    # clear and handed the reader its own process configuration back --
    # the opposite of what was asked for. "This space has no relation
    # meanings" is a thing to say, and it is not the same as saying
    # nothing.
    await engine._emit(space, SAVED,
                       {"space": space,
                        "cleared": meanings is None,
                        "meanings": None if meanings is None else meanings.record()})
    return await read_vocabulary(engine, space)


async def read_vocabulary(engine: "MemoryEngine", space: str) -> Vocabulary:
    """The meanings in force for this space, and where they came from."""
    check_space(space)
    process = engine.relation_meanings
    forgets = None if engine.events is None else _forgets(engine.events)
    if forgets is not None:
        # Said rather than guessed at: this is not "the space holds none",
        # it is "the space cannot hold one here".
        return Vocabulary(
            meanings=process, source="process" if process else "none",
            why=(f"this event log {forgets}, so the space cannot hold a vocabulary of its own; "
                 + ("these are this process's configuration" if process
                    else "and nothing is configured here, so the graph holds only what was said")),
            identity=_identity("process" if process else "none", process, None))
    if engine.events is None:
        return Vocabulary(
            meanings=process, source="process" if process else "none",
            why=("this store keeps no events, so the space can hold no vocabulary of its own; "
                 + ("these are this process's configuration" if process
                    else "and nothing is configured here, so the graph holds only what was said")),
            identity=_identity("process" if process else "none", process, None))

    latest = None
    for event in await engine.events.query(space, kind=SAVED, limit=MAX_READ):
        if event.payload.get("space") == space:
            latest = event
            break
    if latest is None:
        return Vocabulary(
            meanings=process, source="process" if process else "none",
            why=("the space holds no vocabulary, so this process's configuration applies"
                 if process else
                 "the space holds no vocabulary and nothing is configured in this process, so "
                 "the graph holds only what was said"),
            identity=_identity("process" if process else "none", process, None))

    # Read the same way it was written: an explicit clear says so, and an
    # empty vocabulary is a vocabulary. Older records carry no `cleared`
    # key, so a missing `meanings` is read as a clear for those.
    said = latest.payload.get("meanings")
    cleared = bool(latest.payload.get("cleared")) or said is None
    if cleared or not isinstance(said, dict):
        return Vocabulary(
            meanings=process, source="process" if process else "none",
            why=(f"the space's vocabulary was cleared at {latest.ts}, so "
                 + ("this process's configuration applies" if process
                    else "the graph holds only what was said")),
            identity=_identity("process" if process else "none", process, latest.ts),
            at=latest.ts)
    inverse = said.get("inverse")
    symmetric = said.get("symmetric")
    transitive = said.get("transitive")
    held = RelationMeanings(inverse=dict(inverse) if isinstance(inverse, dict) else {},
                            symmetric=list(symmetric) if isinstance(symmetric, list) else (),
                            transitive=list(transitive) if isinstance(transitive, list) else ())
    return Vocabulary(
        meanings=held, source="space",
        why=f"the space holds this vocabulary, saved at {latest.ts}; every reader of the space "
            f"applies it, whatever each process was configured with",
        identity=_identity("space", held, latest.ts), at=latest.ts)
