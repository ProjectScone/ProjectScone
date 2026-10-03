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

This module's own code, run through the engine on the same 3,676 held-out questions
(``benchmarks/public_matched/enginehop.py``), measured hit@1 87.2% (unchanged), all@5 69.8% and all@10 83.2%.
Against LlamaIndex's vector + BM25 fusion it won 348 to 142 at all@5 and 369 to 101 at all@10. Its two recalls
took 424 ms at the median (659 ms p95), against 113 ms for one.

``recall`` is any callable that recalls for a query with the caller's own options, so the second search runs
under the same scope, filters and limits as the first.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.models import RecallItem, RecallResult, RerankTrace
from ..core.validation import MAX_QUERY
from .fusion import PER_EPISODE_CAP

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: How much of each passage a pool reranker reads. A cross-encoder scores a question and a passage within one token
#: window (512 for bge) and refuses a request any pair overflows; on HotpotQA the first 600 characters kept all but
#: 0.7 points of the gain of reading 1,500 (pool_rerank.py) at a third of the time.
PAIR_CHARS = 600

#: Orders a pool of passages for a query; None when the reranker failed, so the caller falls back to its own order.
PoolReranker = Callable[[str, Sequence[RecallItem]], Awaitable[Optional[tuple[list[RecallItem], RerankTrace]]]]

KEEP = 3

#: Words that mark a question comparing two named things ("Which is older, X or Y?", "Are both A and B ..."). Both
#: are named in the question, so the first search finds both, and a hop seeded with one of them only crowds them.
COMPARISON_WORDS = re.compile(r'\b(or|both|either|neither|same)\b', re.IGNORECASE)


def should_hop(question: str) -> bool:
    """Whether a second search is worth running: not for a question that compares things it names.

    Chosen on the development half of HotpotQA from five rules over the question's words alone (the hop then ran on
    78.5% of questions and 15.4% of comparison questions). On the held-out half it kept the hop's gain and removed
    its loss: exact match 46.8% to 49.8% overall (vs 49.4% hopping always), comparison questions 67.6% unchanged (vs
    65.1%). A rule of words, not of meaning: it errs toward not hopping, which costs only the gain it forgoes."""
    return COMPARISON_WORDS.search(question) is None


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


def capped(items: Sequence[RecallItem], limit: int) -> list[RecallItem]:
    """The items in order, at most ``PER_EPISODE_CAP`` of one episode and ``limit`` in all."""
    out: list[RecallItem] = []
    per_episode: dict[int, int] = {}
    for item in items:
        if per_episode.get(item.episode_id, 0) >= PER_EPISODE_CAP:
            continue
        out.append(item)
        per_episode[item.episode_id] = per_episode.get(item.episode_id, 0) + 1
        if len(out) == limit:
            break
    return out


def engine_pool_reranker(engine: "MemoryEngine") -> Optional[PoolReranker]:
    """The engine's own reranker, under its own byte budget and deadline, as a pool reranker; None without one.

    The reranker reads the first ``PAIR_CHARS`` characters of each passage. A failure (timeout, refusal, a broken
    model) returns None, so the caller keeps its own order and the result says no rerank was applied."""
    reranker = engine.reranker
    if reranker is None:
        return None
    from .reranking import MAX_RERANK_LIMIT, RerankCandidate, rerank_candidates

    async def rank(query: str, pool: Sequence[RecallItem]) -> Optional[tuple[list[RecallItem], RerankTrace]]:
        if not pool:
            return None
        candidates = [RerankCandidate(item.chunk_id, item.episode_id, item.text[:PAIR_CHARS], item.source,
                                      item.created_at, item.score, item.similarity, tuple(item.lanes.items()))
                      for item in pool]
        outcome = await rerank_candidates(reranker, query, candidates, limit=min(len(candidates), MAX_RERANK_LIMIT),
                                          max_bytes=engine.rerank_max_bytes, timeout=engine.rerank_timeout)
        if outcome.failure is not None or outcome.trace.status != 'applied':
            return None
        by_id = {item.chunk_id: item for item in pool}
        ordered = [by_id[key] for key in outcome.ordered_ids if key in by_id]
        placed = {item.chunk_id for item in ordered}
        return ordered + [item for item in pool if item.chunk_id not in placed], outcome.trace

    return rank


async def recall_with_hop(recall: Callable[[str], Awaitable[RecallResult]], query: str, *, limit: int,
                          keep: int = KEEP, gate: bool = True,
                          rerank_pool: Optional[PoolReranker] = None) -> tuple[RecallResult, Optional[HopTrace]]:
    """The first recall, extended by a second seeded with its leading passage; the trace is None when no second
    search ran (nothing found, the question leaves no room for a seed within ``MAX_QUERY``, or ``gate`` is on and
    ``should_hop`` says the question compares things it names).

    With ``rerank_pool``, the first search's passages and the hop's are pooled and the reranker orders the pool,
    which replaces the rule-based merge; when no hop runs, the first search alone is reranked. The result's
    ``rerank`` holds the reranker's trace. If the reranker fails, the merge (or the first search's order) stands.

    On HotpotQA's held-out half the hop pool reranked by 8-bit bge-reranker-base put both gold paragraphs in the top
    five for 79.9% of questions against 70.3% for the merge, and lifted single-pass exact match from 51.8% to 55.3%,
    at 369 ms of reranking (benchmarks/public_matched/AGENT_RESULTS.md)."""
    if keep < 1:
        raise ValueError('keep must be at least 1')
    first = await recall(query)
    room = MAX_QUERY - len(query) - 1
    if not first.items or room <= 0 or (gate and not should_hop(query)):
        if rerank_pool is None or not first.items:
            return first, None
        ranked = await rerank_pool(query, first.items)
        if ranked is None:
            return first, None
        items = capped(ranked[0], limit)
        return first.model_copy(update={'items': items, 'rerank': ranked[1],
                                        'returned_bytes': sum(len(item.text.encode()) for item in items)}), None
    seed = first.items[0]
    hop = await recall(query + '\n' + seed.text[:room])
    items = merge(first.items, hop.items, keep=keep, limit=limit)
    rerank_trace = first.rerank
    if rerank_pool is not None:
        pool = list({item.chunk_id: item for item in [*first.items, *hop.items]}.values())
        ranked = await rerank_pool(query, pool)
        if ranked is not None:
            items = capped(ranked[0], limit)
            rerank_trace = ranked[1]
    first_ids = {item.chunk_id for item in first.items}
    trace = HopTrace(seed_chunk_id=seed.chunk_id, seed_chars=min(len(seed.text), room), hop_items=len(hop.items),
                     added=sum(1 for item in items if item.chunk_id not in first_ids))
    merged = first.model_copy(update={
        'items': items,
        'rerank': rerank_trace,
        'returned_bytes': sum(len(item.text.encode()) for item in items),
        'degraded': sorted(set(first.degraded) | set(hop.degraded)),
    })
    return merged, trace
