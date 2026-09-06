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
