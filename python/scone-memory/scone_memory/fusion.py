"""Turn two ranked lists into one, and make the order reproducible.

Reciprocal rank fusion needs no score calibration between lanes, which
matters when one lane is cosine and the other is BM25 or a database's
text score. A small recency term breaks near-ties toward newer memory.
Final scores are normalised so the top item is 1.0: they say how an
item ranks within this query, not how relevant it is in absolute terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .timeutil import parse_rfc3339

RRF_K = 60
W_RECENCY = 0.005
RECENCY_HALF_LIFE_DAYS = 30.0
PER_EPISODE_CAP = 2


@dataclass
class Fused:
    chunk_id: int
    score: float
    similarity: Optional[float]


def rrf(lanes: Sequence[Sequence[tuple[int, float]]], weights: Sequence[float] | None = None) -> dict[int, float]:
    weights = weights or [1.0] * len(lanes)
    scores: dict[int, float] = {}
    for lane, weight in zip(lanes, weights):
        for rank, (chunk_id, _) in enumerate(lane):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank + 1)
    return scores


def recency_boost(created_at: str, now: str) -> float:
    age_days = max(0.0, (parse_rfc3339(now) - parse_rfc3339(created_at)).total_seconds() / 86400)
    return W_RECENCY * math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)


def order(items: list[Fused]) -> list[Fused]:
    """Score descending, then chunk id ascending, so equal scores come
    out the same way on every run and every machine."""
    return sorted(items, key=lambda f: (-f.score, f.chunk_id))


def normalise(items: list[Fused]) -> list[Fused]:
    if not items:
        return items
    top = items[0].score
    if top <= 0:
        return items
    return [Fused(f.chunk_id, f.score / top, f.similarity) for f in items]


MIN_SHARED_WORDS = 4
MIN_SHARED_SHARE = 0.6


def words_of(text: str) -> list[str]:
    return [w for w in text.strip().lower().split() if w]


def restates(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether two texts are the same statement with a different ending:
    "the author of The Marriage of Figaro is Pierre Beaumarchais." and the
    same sentence ending in Thomas Kyd. They must share a prefix of at
    least four words covering at least 60% of the shorter text, so a
    shared opening ("the meeting is ...") does not make two statements one,
    and a value of several words is still caught."""
    if not a or not b or a == b:
        return False
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    return shared >= MIN_SHARED_WORDS and shared >= MIN_SHARED_SHARE * min(len(a), len(b))


def demote_restated(
    items: list[Fused], text_of: dict[int, str], time_of: dict[int, str]
) -> list[Fused]:
    """Within one result, when chunks restate each other (see restates),
    keep the newest at the group's best rank and put the others after it,
    newest first.

    Retrieval otherwise ranks a superseded statement and its replacement
    by embedding similarity, which is nearly equal because they differ
    only at the end, so insertion order decides. Measured on
    MemoryAgentBench Conflict Resolution (E34): the older statement
    outranked the newer in 72 of the 74 single-hop questions that had one.
    Nothing is dropped, since a caller may be asking about the past; only
    the order changes."""
    words = {item.chunk_id: words_of(text_of.get(item.chunk_id, "")) for item in items}
    group_of: dict[int, int] = {}
    groups: list[list[int]] = []
    for position, item in enumerate(items):
        for index, group in enumerate(groups):
            if restates(words[item.chunk_id], words[items[group[0]].chunk_id]):
                group.append(position)
                group_of[position] = index
                break
        else:
            group_of[position] = len(groups)
            groups.append([position])
    ordered = list(items)
    for group in groups:
        if len(group) < 2:
            continue
        members = [items[p] for p in group]
        members.sort(key=lambda f: (time_of.get(f.chunk_id, ""), f.chunk_id), reverse=True)
        for position, member in zip(group, members):
            ordered[position] = member
    return ordered


def cap_per_episode(items: list[Fused], episode_of: dict[int, int], cap: int = PER_EPISODE_CAP) -> list[Fused]:
    seen: dict[int, int] = {}
    kept: list[Fused] = []
    for item in items:
        episode = episode_of.get(item.chunk_id)
        if episode is None:
            continue
        if seen.get(episode, 0) >= cap:
            continue
        seen[episode] = seen.get(episode, 0) + 1
        kept.append(item)
    return kept
