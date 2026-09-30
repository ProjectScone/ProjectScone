"""A second search, seeded by what the first one found, for answers that sit one step away.

A question can name one thing while its answer is in a passage about another thing that the first passage names:
"Where was the director of film X born?" finds the film's page, and the birthplace is on the director's page,
which shares almost no words with the question. Searching again with the question joined to the leading passage
puts the bridge name into the query (multi-hop dense retrieval does this with a trained model; this uses the
engine's own recall).

The second search must not displace what the first search got right. On HotpotQA, fusing the two lists with equal
weights lowered hit@1 from 87.1% to 81.0%. So the first search's leading ``KEEP`` episodes are kept whole and in
order, and the remaining places are filled in turn from the second search and from the rest of the first,
skipping passages already placed and never passing ``PER_EPISODE_CAP`` passages of one episode.

That rule was chosen on one half of the HotpotQA dev set and scored once on the other (3,676 questions, with
documents folded from passages; ``benchmarks/public_matched/hop_rule.py``):

| Metric | Before | After |
| --- | ---: | ---: |
| hit@1 | 87.2% | 87.2% |
| all@2 | 37.3% | 37.3% |
| all@5 | 64.8% | 69.8% |
| all@10 | 76.4% | 83.1% |

It costs one more recall, and a small loss on comparison questions (all@5 91.3% to 88.8%).

``recall`` is any callable that recalls for a query with the caller's own options, so the second search runs
under the same scope, filters and limits as the first.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from ..core.models import RecallItem, RecallResult
from ..core.validation import MAX_QUERY
from .fusion import PER_EPISODE_CAP

KEEP = 3


@dataclass(frozen=True)
class HopTrace:
    """What the second search did: the passage that seeded it, how much of it fit, and what it added."""

    seed_chunk_id: int
    seed_chars: int
    hop_items: int
    added: int


def merge(first: Sequence[RecallItem], hop: Sequence[RecallItem], *, keep: int, limit: int) -> list[RecallItem]:
    """The first search's items through its ``keep``-th distinct episode, then one each from the hop and the rest
    of the first in turn, without repeating a passage or passing ``PER_EPISODE_CAP`` passages of one episode."""
    episodes: list[int] = []
    cut = 0
    for cut, item in enumerate(first, 1):
        if item.episode_id not in episodes:
            if len(episodes) == keep:
                cut -= 1
                break
            episodes.append(item.episode_id)
    else:
        cut = len(first)
    out = list(first[:cut])[:limit]
    placed = {item.chunk_id for item in out}
    per_episode: dict[int, int] = {}
    for item in out:
        per_episode[item.episode_id] = per_episode.get(item.episode_id, 0) + 1
    pools = [list(hop), list(first[cut:])]
    while len(out) < limit and any(pools):
        for pool in pools:
            while pool and (pool[0].chunk_id in placed or per_episode.get(pool[0].episode_id, 0) >= PER_EPISODE_CAP):
                pool.pop(0)
            if pool and len(out) < limit:
                item = pool.pop(0)
                out.append(item)
                placed.add(item.chunk_id)
                per_episode[item.episode_id] = per_episode.get(item.episode_id, 0) + 1
    return out


async def recall_with_hop(recall: Callable[[str], Awaitable[RecallResult]], query: str, *, limit: int,
                          keep: int = KEEP) -> tuple[RecallResult, Optional[HopTrace]]:
    """The first recall, extended by a second seeded with its leading passage; the trace is None when no second
    search ran (nothing found, or the question leaves no room for a seed within ``MAX_QUERY``)."""
    if keep < 1:
        raise ValueError('keep must be at least 1')
    first = await recall(query)
    room = MAX_QUERY - len(query) - 1
    if not first.items or room <= 0:
        return first, None
    seed = first.items[0]
    hop = await recall(query + '\n' + seed.text[:room])
    items = merge(first.items, hop.items, keep=keep, limit=limit)
    first_ids = {item.chunk_id for item in first.items}
    trace = HopTrace(seed_chunk_id=seed.chunk_id, seed_chars=min(len(seed.text), room), hop_items=len(hop.items),
                     added=sum(1 for item in items if item.chunk_id not in first_ids))
    merged = first.model_copy(update={
        'items': items,
        'returned_bytes': sum(len(item.text.encode()) for item in items),
        'degraded': sorted(set(first.degraded) | set(hop.degraded)),
    })
    return merged, trace
