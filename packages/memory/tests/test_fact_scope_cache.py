"""Legacy fact scans reuse bounded source decisions only within one query."""

import asyncio
from collections import Counter

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.ports import NewFact, TextFilter
from scone_memory.retrieval.filters import Condition

WHEN = "2026-09-01T00:00:00Z"


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=lambda: WHEN).open()
    yield engine
    await engine.close()


async def source(memory, team="blue", label="source"):
    added = await memory.remember("alpha", label, kind="file", source=f"manuals/{team}", tags=[team],
        metadata={"team": team}, created_at="2026-01-01T00:00:00Z")
    return await memory.documents.get_episode("alpha", added.episode_id)


async def fact(memory, episode_id, **changes):
    options = dict(space="alpha", subject="needle", predicate="records", object="policy",
                   valid_from="2025-01-01T00:00:00Z", source_episode_id=episode_id)
    return await memory.documents.insert_fact(NewFact(**(options | changes)))


def count_sources(memory, monkeypatch):
    calls = Counter()
    original = memory.documents.get_episode

    async def measured(space, episode_id):
        calls[episode_id] += 1
        return await original(space, episode_id)

    monkeypatch.setattr(memory.documents, "get_episode", measured)
    return calls


async def test_thousand_matching_facts_read_each_of_two_sources_once(memory, monkeypatch):
    allowed = await source(memory, "blue", "allowed source")
    blocked = await source(memory, "red", "blocked source")
    facts = [await fact(memory, allowed.episode_id if index % 10 == 0 else blocked.episode_id)
             for index in range(1000)]
    calls = count_sources(memory, monkeypatch)
    result = await memory._scan_facts_for_query("alpha", "needle policy", WHEN, scope=TextFilter(where={"team": "blue"}))
    assert result == facts[::10][:10]
    assert calls == {allowed.episode_id: 1, blocked.episode_id: 1}


@pytest.mark.parametrize("scope", [
    TextFilter(where={"team": "blue"}), TextFilter(tags=("blue",)), TextFilter(source_prefix="manuals/blue"),
    TextFilter(kind="file", where={"team": "blue"}), TextFilter(conditions=Condition("team", "is", "blue")),
    TextFilter(since="2025-12-31T00:00:00Z", until="2026-01-02T00:00:00Z", where={"team": "blue"}),
    TextFilter(as_of="2026-01-02T00:00:00Z", where={"team": "blue"}),
])
async def test_cached_decisions_preserve_every_existing_source_filter(memory, monkeypatch, scope):
    allowed = await source(memory, "blue", "allowed source")
    blocked = await source(memory, "red", "blocked source")
    wanted = [await fact(memory, allowed.episode_id, confidence=.8) for _ in range(3)]
    for _ in range(3):
        await fact(memory, blocked.episode_id, confidence=1.0)
    calls = count_sources(memory, monkeypatch)
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=scope) == wanted
    assert calls == {allowed.episode_id: 1, blocked.episode_id: 1}


async def test_fact_time_status_overlap_and_ranking_remain_individual(memory, monkeypatch):
    episode = await source(memory)
    active = await fact(memory, episode.episode_id, confidence=.5)
    closed = await fact(memory, episode.episode_id, status="closed", valid_until="2027-01-01T00:00:00Z", confidence=.9)
    await fact(memory, episode.episode_id, status="proposed")
    await fact(memory, episode.episode_id, excluded_reason="suppressed")
    await fact(memory, episode.episode_id, valid_from="2027-01-01T00:00:00Z")
    await fact(memory, episode.episode_id, status="closed", valid_until="2026-01-01T00:00:00Z")
    await fact(memory, episode.episode_id, subject="unrelated", predicate="unrelated", object="unrelated")
    calls = count_sources(memory, monkeypatch)
    assert await memory._scan_facts_for_query("alpha", "needle policy", WHEN, scope=TextFilter(kind="file")) == [closed, active]
    assert calls == {episode.episode_id: 1}


async def test_changed_denied_and_deleted_sources_are_read_fresh_on_each_query(memory, monkeypatch):
    episode = await source(memory)
    expected = [await fact(memory, episode.episode_id) for _ in range(3)]
    scope = TextFilter(where={"team": "blue"})
    calls = count_sources(memory, monkeypatch)
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=scope) == expected
    episode.metadata["team"] = "red"
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=scope) == []
    episode.metadata["team"] = "blue"
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=scope) == expected
    await memory.documents.delete_episode("alpha", episode.episode_id)
    calls.clear()
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=scope) == []
    assert calls == {episode.episode_id: 1}


async def test_lru_cache_is_bounded_and_refreshes_recent_source_access(memory, monkeypatch):
    episodes = [await source(memory, label=f"Source {index}") for index in range(257)]
    for episode in episodes[:256]:
        await fact(memory, episode.episode_id)
    await fact(memory, episodes[0].episode_id)
    await fact(memory, episodes[256].episode_id)
    await fact(memory, episodes[1].episode_id)
    await fact(memory, episodes[0].episode_id)
    calls = count_sources(memory, monkeypatch)
    await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=TextFilter(kind="file"))
    assert sum(calls.values()) == 258
    assert calls[episodes[0].episode_id] == 1 and calls[episodes[1].episode_id] == 2


async def test_stopwords_unscoped_and_unsourced_behavior_stays_unchanged(memory, monkeypatch):
    episode = await source(memory)
    manual = await fact(memory, None)
    sourced = await fact(memory, episode.episode_id)
    calls = count_sources(memory, monkeypatch)
    assert await memory._scan_facts_for_query("alpha", "the and which", WHEN, scope=TextFilter(kind="file")) == []
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN) == [manual, sourced]
    assert calls == {}
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=TextFilter()) == [sourced]
    assert calls == {episode.episode_id: 1} and sourced.quote is None


async def test_cancelled_source_read_propagates_and_does_not_poison_next_query(memory, monkeypatch):
    episode = await source(memory)
    expected = [await fact(memory, episode.episode_id) for _ in range(3)]
    original = memory.documents.get_episode

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(memory.documents, "get_episode", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=TextFilter(kind="file"))
    monkeypatch.setattr(memory.documents, "get_episode", original)
    calls = count_sources(memory, monkeypatch)
    assert await memory._scan_facts_for_query("alpha", "needle", WHEN, scope=TextFilter(kind="file")) == expected
    assert calls == {episode.episode_id: 1}
