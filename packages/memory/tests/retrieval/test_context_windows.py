"""Neighbor passages preserve source identity and ranked-anchor priority."""
import asyncio
import hashlib
import json

import pytest

from scone_memory.core.models import RecallItem, RecallResult
from scone_memory.realtime.context import MemoryContext
from ..retrieval.test_chunk_window import document


async def recalled(engine, monkeypatch, *, source='manual', metadata=None):
    added, chunks = await document(engine, source=source, metadata=metadata or {})
    episode = await engine.documents.get_episode('alpha', added.episode_id)
    seed = chunks[3]
    item = RecallItem(chunk_id=seed.chunk_id, episode_id=seed.episode_id,
        text=seed.text, source=source, created_at=seed.created_at, score=1.,
        metadata=episode.metadata)

    async def recall(*args, **kwargs):
        return RecallResult(items=[item])

    monkeypatch.setattr(engine, 'recall', recall)
    return chunks, item


async def test_windows_recover_adjacent_text_with_real_ids_and_fingerprints(engine, monkeypatch):
    chunks, seed = await recalled(engine, monkeypatch)
    messages = [{'role':'user', 'content':'Which calibration record follows this section?'}]
    baseline, before = await MemoryContext(engine, 'alpha', 'current', limit=1).prepare(messages)
    expanded, after = await MemoryContext(engine, 'alpha', 'current', limit=1, neighbor_chunks=1).prepare(messages)
    assert [r['chunk_id'] for r in before['references']] == [seed.chunk_id]
    assert {r['chunk_id'] for r in after['references']} == {c.chunk_id for c in chunks[2:5]}
    payload = json.loads(expanded[0]['content'].split('\n', 1)[1])
    assert payload['sources'][0]['chunk_id'] == seed.chunk_id
    for source in payload['sources']:
        original = next(c for c in chunks if c.chunk_id == source['chunk_id'])
        assert source['text'] == original.text
        assert after['evidence_fingerprints'][str(original.chunk_id)] == hashlib.sha256(original.text.encode()).hexdigest()
    assert after['context_bytes'] <= 8000 and after['omitted_count'] == 0
    assert after['passage_window_status'] == 'prepared'
    assert after['passage_window_retained_count'] == 2
    neighbor_edges = [edge for edge in after['evidence_graph']['edges']
        if edge['kind'] == 'returned' and edge['target'] != f'chunk:{seed.chunk_id}']
    assert len(neighbor_edges) == 2
    assert all(edge['label'] == 'Read near retrieved passage' for edge in neighbor_edges)
    assert baseline[-1] == expanded[-1] == messages[-1]


async def test_windows_do_not_read_current_session_or_wrong_scope(engine, monkeypatch):
    await recalled(engine, monkeypatch, metadata={'session_id':'current'})

    async def forbidden(*args, **kwargs):
        raise AssertionError('current-session windows must never be requested')

    monkeypatch.setattr(engine.documents, 'page_chunks', forbidden)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}])
    assert receipt['references'] == []


async def test_window_failure_preserves_ranked_anchor(engine, monkeypatch):
    _, seed = await recalled(engine, monkeypatch)

    async def unavailable(*args, **kwargs):
        raise RuntimeError('storage unavailable')

    monkeypatch.setattr(engine.documents, 'page_chunks', unavailable)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', limit=1, neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}])
    assert [r['chunk_id'] for r in receipt['references']] == [seed.chunk_id]
    assert receipt['passage_window_status'] == 'unavailable'
    assert receipt['passage_window_retained_count'] == 0


async def test_window_timeout_preserves_ranked_anchor(engine, monkeypatch):
    _, seed = await recalled(engine, monkeypatch)

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(engine.documents, 'page_chunks', slow)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', limit=1,
        neighbor_chunks=1, recall_timeout=.01).prepare([{'role':'user', 'content':'calibration records'}])
    assert [r['chunk_id'] for r in receipt['references']] == [seed.chunk_id]
    assert receipt['passage_window_status'] == 'timeout'


async def test_wrong_scope_recall_cannot_expand_or_survive_verification(engine, monkeypatch):
    await recalled(engine, monkeypatch, metadata={'team':'private'})
    _, receipt = await MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1,
        where={'team':'public'}).prepare([{'role':'user', 'content':'calibration records'}])
    assert receipt['references'] == [] and receipt['passage_window_retained_count'] == 0


async def test_deleted_during_window_read_cannot_reach_context(engine, monkeypatch):
    _, seed = await recalled(engine, monkeypatch)
    original = engine.documents.page_chunks

    async def delete_after_read(*args, **kwargs):
        rows = await original(*args, **kwargs)
        await engine.forget('alpha', seed.episode_id)
        return rows

    monkeypatch.setattr(engine.documents, 'page_chunks', delete_after_read)
    request, receipt = await MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}])
    assert len(request) == 1 and receipt['references'] == []


async def test_byte_budget_keeps_anchor_and_omits_whole_neighbors(engine, monkeypatch):
    _, seed = await recalled(engine, monkeypatch)
    messages = [{'role':'user', 'content':'calibration records'}]
    _, baseline = await MemoryContext(engine, 'alpha', 'current', limit=1).prepare(messages)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', limit=1,
        neighbor_chunks=4, max_context_bytes=max(512, baseline['context_bytes'])).prepare(messages)
    assert receipt['references'] == baseline['references']
    assert receipt['context_bytes'] <= max(512, baseline['context_bytes'])
    assert receipt['omitted_count'] == receipt['passage_window_candidate_count']


async def test_two_ranked_anchors_stay_first_and_shared_neighbors_are_unique(engine, monkeypatch):
    chunks, seed = await recalled(engine, monkeypatch)
    second = seed.model_copy(update={'chunk_id':chunks[5].chunk_id, 'text':chunks[5].text})

    async def recall(*args, **kwargs):
        return RecallResult(items=[seed, second])

    monkeypatch.setattr(engine, 'recall', recall)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', limit=2, neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}])
    ids = [item['chunk_id'] for item in receipt['references']]
    assert ids[:2] == [seed.chunk_id, second.chunk_id]
    assert len(ids) == len(set(ids)) == 5
    assert receipt['passage_window_anchors'][str(chunks[4].chunk_id)] == [seed.chunk_id, second.chunk_id]


async def test_adaptive_selection_does_not_gain_unassessed_neighbors(engine, monkeypatch):
    from ..retrieval.test_adaptive_context import ScriptedAssessor, retriever, select_all
    _, seed = await recalled(engine, monkeypatch)

    async def forbidden(*args, **kwargs):
        raise AssertionError('adaptive selection must not read neighbors')

    monkeypatch.setattr(engine.documents, 'page_chunks', forbidden)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1,
        adaptive_retriever=retriever(engine, ScriptedAssessor(select_all))).prepare(
        [{'role':'user', 'content':'calibration records'}])
    assert [r['chunk_id'] for r in receipt['references']] == [seed.chunk_id]
    assert receipt['passage_window_status'] == 'skipped'


async def test_text_conversation_delivers_window_passages_to_model(engine, monkeypatch):
    from scone_memory.realtime.text import TextConversation
    from ..conversations.test_text_conversation import ScriptedModel
    chunks, _ = await recalled(engine, monkeypatch)
    model = ScriptedModel()
    conversation = TextConversation(engine, 'alpha', 'current', lambda: model, neighbor_chunks=1)
    result = await conversation.reply('Which calibration record follows this section?')
    assert result['memory_context']['passage_window_retained_count'] == 2
    payload = next(message['content'] for message in model.requests[0] if 'Scone retrieved' in message['content'])
    assert chunks[4].text in payload


async def test_optional_graph_failure_drops_unverified_windows(engine, monkeypatch):
    import scone_memory.realtime.context as module
    _, seed = await recalled(engine, monkeypatch)

    async def unavailable(*args, **kwargs):
        raise RuntimeError('graph unavailable')

    monkeypatch.setattr(module, 'build_query_evidence_graph', unavailable)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}])
    assert [r['chunk_id'] for r in receipt['references']] == [seed.chunk_id]
    assert receipt['passage_window_retained_count'] == 0
    assert receipt['passage_window_status'] == 'unavailable'


async def test_window_cancellation_propagates(engine, monkeypatch):
    await recalled(engine, monkeypatch)
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(engine.documents, 'page_chunks', blocked)
    pending = asyncio.create_task(MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}]))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


async def test_window_expansion_caps_total_sources_without_losing_twenty_anchors(engine, monkeypatch):
    engine.chunk_target = 100
    added = await engine.remember('alpha', '\n\n'.join(
        f'Record {i}: '+ 'calibration sample ' * 10 for i in range(100)), source='manual')
    chunks = await engine.documents.chunks_of('alpha', added.episode_id)
    seeds = [RecallItem(chunk_id=c.chunk_id, episode_id=c.episode_id, text=c.text,
        created_at=c.created_at, source='manual', score=1.) for c in chunks[::3][:20]]
    assert len(seeds) == 20

    async def recall(*args, **kwargs):
        return RecallResult(items=seeds)

    monkeypatch.setattr(engine, 'recall', recall)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', limit=20,
        max_context_bytes=64000, neighbor_chunks=4).prepare([{'role':'user', 'content':'calibration records'}])
    ids = [r['chunk_id'] for r in receipt['references']]
    assert ids[:20] == [seed.chunk_id for seed in seeds]
    assert len(ids) == len(set(ids)) == 24


async def test_revision_change_between_windows_and_graph_discards_snapshot(engine, monkeypatch):
    import scone_memory.realtime.context as module
    await recalled(engine, monkeypatch)
    original = module.passage_windows

    async def mutate_after_read(*args, **kwargs):
        result = await original(*args, **kwargs)
        await engine.remember('alpha', 'Another record was added concurrently.')
        return result

    monkeypatch.setattr(module, 'passage_windows', mutate_after_read)
    _, receipt = await MemoryContext(engine, 'alpha', 'current', neighbor_chunks=1).prepare(
        [{'role':'user', 'content':'calibration records'}])
    assert receipt['references'] == [] and receipt['evidence_graph_stale'] is True


@pytest.mark.parametrize('radius', [-1, 5, True, 1.5, '1'])
async def test_window_radius_is_bounded(engine, radius):
    with pytest.raises(ValueError, match='neighbor_chunks'):
        MemoryContext(engine, 'alpha', 'current', neighbor_chunks=radius)
