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

from ..core.errors import InvalidInput
from ..core.timeutil import parse_rfc3339

RRF_K = 60
#: The recency term's size at age zero, and the age at which it has
#: halved. Small against a fused rank score so it breaks near-ties only;
#: an engine can be built with its own (a memory of a life is not a
#: memory of a build log), and zero turns it off.
W_RECENCY = 0.005
RECENCY_HALF_LIFE_DAYS = 30.0
MAX_RECENCY_WEIGHT = 1.0
MAX_RECENCY_HALF_LIFE_DAYS = 36_500.0
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


def validate_recency(weight: float, half_life_days: float) -> None:
    """A weight is zero or more, at most MAX_RECENCY_WEIGHT; a half-life is
    a positive number of days, at most a century. Both finite."""
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= MAX_RECENCY_WEIGHT:
        raise InvalidInput(f"recency_weight must be a finite number from 0 to {MAX_RECENCY_WEIGHT}")
    if (isinstance(half_life_days, bool) or not isinstance(half_life_days, (int, float)) or not math.isfinite(half_life_days)
            or not 0 < half_life_days <= MAX_RECENCY_HALF_LIFE_DAYS):
        raise InvalidInput(f"recency_half_life_days must be finite and greater than zero, at most {MAX_RECENCY_HALF_LIFE_DAYS:g}")


def recency_boost(created_at: str, now: str, *, weight: float = W_RECENCY,
                  half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """How much newer memory is favoured: ``weight`` at age zero, halved
    every ``half_life_days``. Zero weight favours nothing."""
    if weight == 0:
        return 0.0
    age_days = max(0.0, (parse_rfc3339(now) - parse_rfc3339(created_at)).total_seconds() / 86400)
    # A half-life halves: at age half_life_days the term is weight / 2.
    # (The constant this replaced was a 1/e time constant under the same
    # name, which halved at 0.69 of the days it named.)
    return weight * 2.0 ** (-age_days / half_life_days)


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
