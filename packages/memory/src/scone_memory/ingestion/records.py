"""Ingestion records, recovery receipts and source identity helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.ports import NewEpisode


@dataclass(frozen=True)
class Record:
    """One thing to remember, for batch ingest and for import."""

    content: str
    kind: str = "note"
    source: Optional[str] = None
    tags: Sequence[str] = ()
    created_at: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    #: Identity for deduplication. By default two records with the same
    #: text in a space are one episode. A transcript is different: the
    #: second "ok" in a conversation is a second turn, so a chat adapter
    #: keys each turn ("<session>#<n>") and identical text under different
    #: keys is stored twice, while a retried write of the same key is not.
    dedup_key: Optional[str] = None
    #: The identity a dump carries. Import passes it through so a moved
    #: store deduplicates exactly as its source did; callers leave it None.
    content_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping) -> "Record":
        known = {k: data[k] for k in ("content", "kind", "source", "tags", "created_at", "metadata", "dedup_key", "content_hash") if k in data}
        if "content" not in known:
            raise InvalidInput("a record needs content")
        return cls(**known)


@dataclass
class RecoveryReport:
    """What recover() found: episodes brought to a complete state, of
    which how many needed their chunks rebuilt, and marks with no
    episode behind them (the write never landed). ``retired`` counts completed
    pending source deletions. If ``retirements_pending`` is true, the bounded
    cleanup pass left work and ingestion repair has not run yet."""

    completed: int = 0
    rechunked: int = 0
    forgotten: int = 0
    retired: int = 0
    retirements_pending: bool = False


@dataclass(frozen=True)
class _Pending:
    slot: int
    new: NewEpisode
    #: Chunk texts, sliced in code points.
    texts: list[str]
    #: The same spans as UTF-8 byte offsets (spec rule 1.2).
    spans: list[tuple[int, int]]


@dataclass(frozen=True)
class _DupOf:
    """A record identical to an earlier one in the same batch; its id is
    known only after that one is written."""

    slot: int


def content_hash(space: str, content: str, dedup_key: Optional[str] = None) -> str:
    if dedup_key is not None:
        return key_hash(space, dedup_key)
    return hashlib.sha256(f"{space}\x00{content.strip()}".encode()).hexdigest()


def key_hash(space: str, dedup_key: str) -> str:
    """The same exact, space-bound identity for keyed reads and writes."""
    if not isinstance(dedup_key, str) or not 1 <= len(dedup_key) <= 256:
        raise InvalidInput("dedup_key must be a string of 1..=256 chars")
    try:
        encoded = f"{space}\x00key\x00{dedup_key}".encode("utf-8")
    except UnicodeError as error:
        raise InvalidInput("dedup_key must be valid UTF-8") from error
    return hashlib.sha256(encoded).hexdigest()


def contextual_prefix(episode: "NewEpisode") -> str:
    """The context a chunk loses when cut from its episode: when it
    happened, where it came from, and whose it is. Prepended to the text
    that is embedded (research experiment 8, Anthropic's contextual
    retrieval without the LLM), never to the text that is stored, so
    invariant I1 holds and recall still returns exact excerpts. Only
    fields that exist are named; an empty prefix means no change."""
    parts = []
    if episode.created_at:
        parts.append(episode.created_at[:10])
    if episode.source:
        parts.append(episode.source)
    for key in ("user_id", "agent_id", "session_id"):
        value = episode.metadata.get(key) if episode.metadata else None
        if value:
            parts.append(f"{key.replace('_', ' ')} {value}")
    return " | ".join(parts)
