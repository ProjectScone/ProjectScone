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
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .models import IngestJob, JobItem, Chunk, Episode, Fact, FactLink, Tombstone


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
class NewTombstone:
    space: str
    episode_id: int
    content_hash: str
    forgotten_at: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class NewFactLink:
    space: str
    from_fact: int
    to_fact: int
    kind: str
    created_at: str
    source_episode_id: Optional[int] = None
    quote: Optional[str] = None


@dataclass(frozen=True)
class TextFilter:
    """What the lexical lane may return. ``as_of`` excludes anything
    that happened after that instant; ``tags`` must all be present."""

    as_of: Optional[str] = None
    tags: tuple[str, ...] = ()
    #: Every key must match the episode's metadata exactly.
    where: Mapping[str, str] = field(default_factory=dict)
    #: A parsed metadata filter, or None. Duck-typed rather than imported
    #: so this layer keeps depending on nothing above it: it answers
    #: ``matches(metadata)`` and renders itself with ``to_sql(column)``.
    #: A store that ignores it still answers correctly, because the
    #: engine checks every candidate again; it just looks at more of them.
    conditions: Optional[Any] = None


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


@dataclass(frozen=True)
class DeletedSpace:
    """What a document store removed for ``delete_space``: the chunk ids
    (so the vector index can follow) and the counts for the receipt."""

    chunk_ids: tuple[int, ...]
    episodes: int
    facts: int
    links: int
    tombstones: int


@dataclass(frozen=True)
class NewJob:
    """A batch to record, with the receipts its records already have."""

    job_id: str
    space: str
    created_at: str
    request_id: Optional[str]
    items: tuple[JobItem, ...]


@runtime_checkable
class EpisodeInventory(Protocol):
    """Optional bounded inventory, newest ID first; apply filters before LIMIT.

    Existing custom DocumentStore implementations can omit this operation.
    The engine validates arguments; stores receive limits from 1 through 101.
    """

    async def page_episodes(self, space: str, before: Optional[int], limit: int,
                            kind: Optional[str]) -> list[Episode]: ...


@dataclass(frozen=True)
class SourcePage:
    episodes: list[Episode]
    has_more: bool
    next_before: Optional[int]


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
    async def chunks_of(self, space: str, episode_id: int) -> list[Chunk]:
        """Every chunk of one episode, in ordinal order."""
        ...

    async def mark_inflight(self, space: str, content_hash: str) -> None:
        """Record that a write for this episode identity has started and
        its vectors may not have landed. Cleared by ``clear_inflight``
        once the whole episode (rows and vectors) is durable; anything
        still marked when the engine next opens is repaired first. Marks
        are idempotent per (space, hash)."""
        ...

    async def clear_inflight(self, space: str, content_hash: str) -> None: ...
    async def inflight(self) -> list[tuple[str, str]]:
        """Every (space, content_hash) still marked, any space."""
        ...
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
    async def record_tombstone(self, new: NewTombstone) -> Tombstone:
        """Remember that an episode existed and was forgotten; the same
        (space, episode_id) recorded again returns the first record."""
        ...
    async def tombstone(self, space: str, episode_id: int) -> Optional[Tombstone]: ...
    async def tombstone_by_hash(self, space: str, content_hash: str) -> Optional[Tombstone]: ...
    async def list_tombstones(self, space: str) -> list[Tombstone]: ...
    async def insert_fact_link(self, new: NewFactLink) -> FactLink:
        """Store a link; the same (space, from, to, kind) stored again returns
        the stored link unchanged, so linking is idempotent everywhere."""
        ...
    async def fact_links(self, space: str, fact_id: int) -> list[FactLink]:
        """Every link naming the fact at either end, oldest first."""
        ...
    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        """Every fact, any status, with this subject and predicate."""
        ...

    async def bump_revision(self, space: str) -> int: ...
    async def revision(self, space: str) -> int: ...
    async def delete_space(self, space: str, deleted_at: str) -> DeletedSpace:
        """Remove every record of the space (chunks, episodes, links,
        facts, tombstones, inflight marks, the revision row) and mark it
        deleted at ``deleted_at``; atomic where the store can be."""
        ...
    async def space_deleted(self, space: str) -> Optional[str]:
        """When the space was deleted, or None while it lives."""
        ...
    async def create_job(self, new: NewJob) -> IngestJob:
        """Record a batch. Optional: a store that cannot keep jobs simply
        does not implement this, and the engine says so plainly."""
        ...
    async def get_job(self, space: str, job_id: str) -> Optional[IngestJob]: ...
    async def job_by_request(self, space: str, request_id: str) -> Optional[IngestJob]:
        """The job this request already made, so a retry is not a second one."""
        ...
    async def list_jobs(self, space: str, limit: int, before: Optional[str] = None) -> list[IngestJob]:
        """Newest first. ``before`` names the last job of the previous page,
        so a caller can walk back through older batches."""
        ...
    async def update_job(self, job: IngestJob) -> None: ...
    async def mark_failed(self, space: str, episode_id: int, error: str, when: str) -> int:
        """Record that reading this record failed, counting the attempt.
        A record already read is left alone."""
        ...
    async def mark_consolidated(self, space: str, episode_ids: Sequence[int], when: str) -> int:
        """Record that these episodes have been read; returns how many
        items moved, so marking the same episode twice is not two moves."""
        ...


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
    async def purge(self, space: str, *, preview: bool = False) -> int:
        """How many events the space holds; remove them unless ``preview``."""
        ...
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
