"""Conversation turns as episodes, and recall items as plain records.

A turn is one message of one session. It is stored as its own episode so
that a session reads back in order and a single turn can be popped, with
metadata ``session_id``, ``role`` and ``seq`` (its position, as text
because metadata values are strings). Deduplication is keyed by
``<session>#<seq>``: the second "ok" in a conversation is a second turn.

What the episode's content is depends on the message. A plain text
message is stored as its text, so recall over the transcript reads well.
Anything else (structured content, tool calls, extra fields) is stored
as its JSON with ``encoding=json`` in the metadata, so it comes back
exactly as it went in. Either way the read side is lossless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..memory.engine import Record
from ..core.models import Episode, RecallItem


@dataclass(frozen=True)
class Turn:
    role: str
    #: A plain text message, or None when ``payload`` carries the whole item.
    text: Optional[str]
    #: The verbatim item when it was not a plain text message.
    payload: Optional[Any] = None

    @property
    def content(self) -> str:
        return self.text if self.text is not None else json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


def turn_records(
    session_id: str,
    turns: Sequence[Turn],
    start_seq: int,
    extra: Mapping[str, str] | None = None,
) -> list[Record]:
    """One record per turn, keyed for deduplication by session and position."""
    out = []
    for offset, turn in enumerate(turns):
        seq = start_seq + offset
        metadata = {**(extra or {}), "session_id": session_id, "role": turn.role, "seq": str(seq)}
        if turn.text is None:
            metadata["encoding"] = "json"
        out.append(Record(turn.content, kind="conversation", source=session_id, metadata=metadata, dedup_key=f"{session_id}#{seq}"))
    return out


def read_turn(episode: Episode) -> Turn:
    """The turn an episode holds, exactly as it was stored."""
    role = episode.metadata.get("role", "user")
    if episode.metadata.get("encoding") == "json":
        return Turn(role, None, json.loads(episode.content))
    return Turn(role, episode.content)


def next_seq(episodes: Iterable[Episode]) -> int:
    """One past the highest stored position. Read from the store on every
    write, not counted in memory, so two writers of one session do not
    hand out the same key."""
    return max((int(e.metadata.get("seq", "0")) for e in episodes), default=-1) + 1


def item_metadata(item: RecallItem) -> dict[str, Any]:
    """What a framework document should say about where it came from: the
    episode and chunk, the fusion score and optional reranker score, the cosine when the vector lane saw
    it, the lanes that ranked it, when it happened, and the episode's own
    scope metadata (which cannot shadow these keys)."""
    return {
        **dict(item.metadata),
        "episode_id": item.episode_id,
        "chunk_id": item.chunk_id,
        "score": item.score,
        "rerank_score": item.rerank_score,
        "similarity": item.similarity,
        "lanes": dict(item.lanes),
        "created_at": item.created_at,
        "source": item.source,
        "tags": list(item.tags),
    }
