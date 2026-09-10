"""Typed views of the JSON the Scone HTTP API returns.

Every dataclass mirrors one shape emitted by ``crates/scone/src/serve.rs``.
Optional fields are genuinely optional: ``/v1/profile`` returns a narrower
fact than ``/v1/recall`` does, and a server that grows a field must not
break a client that has not learned about it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Fact", "Memory", "Profile", "Status", "Tag", "Recall", "Added"]

JsonDict = Dict[str, Any]


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


@dataclass(frozen=True)
class Fact:
    """A subject-predicate-object assertion with a validity interval."""

    fact_id: int
    subject: str
    predicate: str
    object: str
    confidence: float
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    status: Optional[str] = None

    @classmethod
    def from_json(cls, data: JsonDict) -> "Fact":
        """Build a fact from one entry of a ``facts`` or ``static_facts`` array."""
        return cls(
            fact_id=int(data["fact_id"]),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            object=str(data.get("object", "")),
            confidence=float(data.get("confidence", 0.0)),
            valid_from=_opt_str(data.get("valid_from")),
            valid_until=_opt_str(data.get("valid_until")),
            status=_opt_str(data.get("status")),
        )


@dataclass(frozen=True)
class Memory:
    """One recalled chunk of episodic text, with its retrieval score."""

    episode_id: int
    text: str
    score: float
    created_at: Optional[str] = None
    source: Optional[str] = None

    @property
    def day(self) -> Optional[str]:
        """The calendar day of ``created_at``, or None if the server omitted it."""
        if self.created_at is None:
            return None
        return self.created_at.split("T", 1)[0]

    @classmethod
    def from_json(cls, data: JsonDict) -> "Memory":
        """Build a memory from one entry of a recall ``items`` array."""
        return cls(
            episode_id=int(data["episode_id"]),
            text=str(data.get("text", "")),
            score=float(data.get("score", 0.0)),
            created_at=_opt_str(data.get("created_at")),
            source=_opt_str(data.get("source")),
        )


@dataclass(frozen=True)
class Recall:
    """The context pack a recall returns: facts first, then episodic memories."""

    facts: List[Fact] = field(default_factory=list)
    items: List[Memory] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    returned_bytes: int = 0
    space_bytes: int = 0
    context_reduction: float = 0.0

    def __iter__(self):
        """Iterate the recalled memories, so ``for m in client.recall(...)`` reads well."""
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def from_json(cls, data: JsonDict) -> "Recall":
        """Build a context pack from the ``GET /v1/recall`` body."""
        return cls(
            facts=[Fact.from_json(f) for f in data.get("facts") or []],
            items=[Memory.from_json(i) for i in data.get("items") or []],
            degraded=[str(d) for d in data.get("degraded") or []],
            returned_bytes=int(data.get("returned_bytes") or 0),
            space_bytes=int(data.get("space_bytes") or 0),
            context_reduction=float(data.get("context_reduction") or 0.0),
        )


@dataclass(frozen=True)
class Added:
    """The outcome of storing an episode, including whether it deduplicated."""

    episode_id: int
    deduplicated: bool = False
    chunks: Optional[int] = None

    @classmethod
    def from_json(cls, data: JsonDict) -> "Added":
        """Build an ingest outcome from the ``POST /v1/episodes`` body.

        A deduplicated store carries no ``chunks``, so it stays None.
        """
        chunks = data.get("chunks")
        return cls(
            episode_id=int(data["episode_id"]),
            deduplicated=bool(data.get("deduplicated", False)),
            chunks=None if chunks is None else int(chunks),
        )


@dataclass(frozen=True)
class Profile:
    """Who the space is about: durable facts plus recent activity lines."""

    static_facts: List[Fact] = field(default_factory=list)
    dynamic: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: JsonDict) -> "Profile":
        """Build a profile from the ``GET /v1/profile`` body."""
        return cls(
            static_facts=[Fact.from_json(f) for f in data.get("static_facts") or []],
            dynamic=[str(d) for d in data.get("dynamic") or []],
        )


@dataclass(frozen=True)
class Status:
    """Size and health of the one space the API key is bound to."""

    space: str
    episodes: int = 0
    chunks: int = 0
    revision: int = 0
    semantic_lane: Optional[str] = None
    pending_distill: int = 0

    @property
    def semantic_lane_active(self) -> bool:
        """True when an LLM is attached and facts are being distilled."""
        return self.semantic_lane == "active"

    @classmethod
    def from_json(cls, data: JsonDict) -> "Status":
        """Build a status from the ``GET /v1/status`` body."""
        return cls(
            space=str(data.get("space", "")),
            episodes=int(data.get("episodes") or 0),
            chunks=int(data.get("chunks") or 0),
            revision=int(data.get("revision") or 0),
            semantic_lane=_opt_str(data.get("semantic_lane")),
            pending_distill=int(data.get("pending_distill") or 0),
        )


@dataclass(frozen=True)
class Tag:
    """A tag name and how many episodes carry it."""

    name: str
    count: int = 0

    @classmethod
    def from_json(cls, data: JsonDict) -> "Tag":
        """Build a tag from one entry of the ``GET /v1/tags`` array."""
        return cls(name=str(data["name"]), count=int(data.get("count") or 0))
