from __future__ import annotations

import asyncio

import pytest

from scone_memory.core.models import RecallItem, RecallResult
from scone_memory.core.validation import MAX_QUERY
from scone_memory.retrieval.second_hop import merge, recall_with_hop


def _item(chunk_id: int, episode_id: int, text: str = '') -> RecallItem:
    return RecallItem(chunk_id=chunk_id, episode_id=episode_id, text=text or f'passage {chunk_id}', score=1.0,
                      created_at='2026-01-01T00:00:00Z')


def test_the_first_searchs_leading_episodes_stay_whole_and_in_order() -> None:
    first = [_item(1, 10), _item(2, 10), _item(3, 20), _item(4, 30), _item(5, 40), _item(6, 50)]
    hop = [_item(1, 10), _item(7, 60), _item(8, 70)]
    merged = merge(first, hop, keep=3, limit=6)
    # episodes 10, 20, 30 kept whole; then the hop's new passage, the first's next, the hop's next
    assert [i.chunk_id for i in merged] == [1, 2, 3, 4, 7, 5]


def test_merge_never_repeats_a_passage_or_passes_the_per_episode_cap() -> None:
    first = [_item(1, 10), _item(2, 20)]
    hop = [_item(2, 20), _item(3, 10), _item(4, 10), _item(5, 30)]
    merged = merge(first, hop, keep=1, limit=10)
    ids = [i.chunk_id for i in merged]
    assert len(ids) == len(set(ids))
    assert sum(1 for i in merged if i.episode_id == 10) <= 2 and 5 in ids


def test_merge_respects_the_limit_even_inside_the_kept_lead() -> None:
    first = [_item(i, i) for i in range(1, 6)]
    assert [i.chunk_id for i in merge(first, [], keep=5, limit=3)] == [1, 2, 3]


def test_the_second_search_is_seeded_with_the_leading_passage_within_the_query_limit() -> None:
    asked: list[str] = []

    async def recall(query: str) -> RecallResult:
        asked.append(query)
        if len(asked) == 1:
            return RecallResult(items=[_item(1, 10, 'Film X was directed by Ann Lee. ' * 80), _item(2, 20)])
        return RecallResult(items=[_item(3, 30, 'Ann Lee was born in Lyon.')], degraded=['text'])

    result, trace = asyncio.run(recall_with_hop(recall, 'Where was the director of film X born?', limit=5))
    assert asked[1].startswith('Where was the director of film X born?\nFilm X was directed by Ann Lee.')
    assert len(asked[1]) == MAX_QUERY
    assert [i.chunk_id for i in result.items] == [1, 2, 3]
    assert trace is not None and trace.seed_chunk_id == 1 and trace.added == 1 and result.degraded == ['text']


def test_no_second_search_without_a_first_result_or_room_for_a_seed() -> None:
    calls = 0

    async def recall(query: str) -> RecallResult:
        nonlocal calls
        calls += 1
        return RecallResult(items=[_item(1, 10)] if calls == 1 and len(query) < MAX_QUERY else [])

    result, trace = asyncio.run(recall_with_hop(recall, 'x' * MAX_QUERY, limit=5))
    assert trace is None and calls == 1
    with pytest.raises(ValueError):
        asyncio.run(recall_with_hop(recall, 'q', limit=5, keep=0))
