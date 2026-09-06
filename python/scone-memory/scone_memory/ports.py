"""The three seams the engine is built across.

Everything environment-specific sits behind one of these: where
documents live, where vectors live, and what turns text into vectors.
The engine only ever talks to the protocols, so the same code runs on a
laptop with dicts, in a container against MongoDB and Qdrant, or in a
test with a deterministic embedder. A new backend implements one
protocol and passes the contract tests; nothing in the engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from .models import Chunk, Episode, Fact


@dataclass(frozen=True)
class NewEpisode:
    space: str
    kind: str
    content: str
    content_hash: str
    created_at: str
    ingested_at: str
    source: Optional[str] = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NewChunk:
    episode_id: int
    space: str
    ordinal: int
    start: int
    end: int
    text: str
    created_at: str


@dataclass(frozen=True)
class NewFact:
    space: str
    subject: str
    predicate: str
    object: str
    valid_from: str
    confidence: float = 1.0
    valid_until: Optional[str] = None
    status: str = "active"
    closed_reason: Optional[str] = None
    source_episode_id: Optional[int] = None
    origin: str = "stated"
    superseded_by: Optional[int] = None
    excluded_reason: Optional[str] = None
    quote: Optional[str] = None


@dataclass(frozen=True)
class TextFilter:
    """What the lexical lane may return. ``as_of`` excludes anything
    that happened after that instant; ``tags`` must all be present."""

    as_of: Optional[str] = None
    tags: tuple[str, ...] = ()
    #: Every key must match the episode's metadata exactly.
    where: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorPoint:
    chunk_id: int
    space: str
    episode_id: int
    created_at: str
    vector: Sequence[float]
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass
class SpaceCounts:
    episodes: int = 0
    chunks: int = 0
    bytes: int = 0
    tags: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class DocumentStore(Protocol):
    """Truth: episodes, their chunks, and facts. Also the lexical lane,
    because full-text search wants to live next to the text."""

    name: str

    async def insert_episode(self, new: NewEpisode) -> Episode: ...
    async def episode_by_hash(self, space: str, content_hash: str) -> Optional[Episode]: ...
    async def get_episode(self, space: str, episode_id: int) -> Optional[Episode]: ...
    async def delete_episode(self, space: str, episode_id: int) -> list[int]:
        """Remove an episode and its chunks; return the chunk ids removed
        so the vector index can follow."""
        ...

    async def insert_chunks(self, new: Sequence[NewChunk]) -> list[Chunk]: ...
    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]: ...
    async def search_text(
        self, space: str, query: str, limit: int, filter: TextFilter
    ) -> list[tuple[int, float]]:
        """Ranked (chunk_id, score), best first. Scores are lane-local;
        only their order is used."""
        ...

    async def recent_episodes(self, space: str, limit: int) -> list[Episode]: ...
    async def counts(self, space: str) -> SpaceCounts: ...

    async def insert_fact(self, new: NewFact) -> Fact: ...
    async def update_fact(self, fact: Fact) -> None: ...
    async def get_fact(self, space: str, fact_id: int) -> Optional[Fact]: ...
    async def list_facts(self, space: str, include_closed: bool) -> list[Fact]: ...
    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        """Every fact, any status, with this subject and predicate."""
        ...

    async def bump_revision(self, space: str) -> int: ...
    async def revision(self, space: str) -> int: ...


@runtime_checkable
class VectorIndex(Protocol):
    name: str

    async def ensure(self, dim: int) -> None:
        """Create or check the collection for vectors of this width."""
        ...

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...
    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        """Ranked (chunk_id, cosine similarity), best first."""
        ...

    async def delete(self, chunk_ids: Sequence[int]) -> None: ...


#: Bumped when an event payload changes shape; readers check it before
#: computing anything from a payload.
EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NewEvent:
    ts: str
    space: str
    kind: str
    payload: Mapping[str, object]
    schema_version: int = EVENT_SCHEMA_VERSION
    #: Connector-supplied identity for an externally reported event, unique
    #: per space. A second append with the same key returns the stored
    #: event when the payload matches and raises DuplicateEvent when it
    #: differs. None for engine events.
    dedup_key: Optional[str] = None


@dataclass(frozen=True)
class Event:
    event_id: int
    ts: str
    space: str
    kind: str
    payload: Mapping[str, object]
    schema_version: int = EVENT_SCHEMA_VERSION
    dedup_key: Optional[str] = None


class DuplicateEvent(Exception):
    """The dedup_key is already stored with a different payload."""

    def __init__(self, existing: "Event") -> None:
        super().__init__(f"event {existing.event_id} already holds dedup key {existing.dedup_key!r} with a different payload")
        self.existing = existing


@runtime_checkable
class EventLog(Protocol):
    """Evidence: one record per engine operation, appended after the
    operation finished or failed. Metrics are computed from these and
    from nothing else."""

    name: str

    async def append(self, new: NewEvent) -> Event:
        """Store and return. With a dedup_key: return the existing event if
        one holds the same key and payload in this space; raise
        DuplicateEvent if the payload differs. Atomic per sink."""
        ...

    async def get(self, space: str, event_id: int) -> Optional[Event]: ...
    async def query(
        self,
        space: str,
        kind: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        after_id: Optional[int] = None,
    ) -> list[Event]:
        """Newest first. ``since`` is inclusive on the event timestamp.

        With ``after_id``, oldest first and only events with a larger id:
        a stable cursor for a reader that must not miss a burst, since
        ids are assigned at receipt and never reused."""
        ...


@runtime_checkable
class Embedder(Protocol):
    #: Names the model; vectors from different ids are never compared.
    id: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
