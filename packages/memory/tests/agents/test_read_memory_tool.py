"""A read tool expands only authorized, retained neighboring chunks."""
import json
from unittest.mock import AsyncMock

import pytest

from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope
from ..retrieval.test_chunk_window import document


def box(engine, **kwargs):
    return ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated(where={'team':'blue'}),
                             exclude_session_id='current', **kwargs)


async def test_read_returns_exact_neighbor_quotes_and_fresh_sources(engine):
    added, chunks = await document(engine, metadata={'team':'blue'})
    tools = box(engine)
    evidence = await tools.prepare('read_memory', {'chunk_id':chunks[4].chunk_id, 'before':1, 'after':2})
    packet = json.loads(evidence.payload)
    assert packet['status'] == 'prepared'
    assert [row['text'] for row in packet['items']] == [row.text for row in chunks[3:7]]
    assert evidence.evidence_ids == tuple(f'chunk:{row.chunk_id}' for row in chunks[3:7])
    assert all(row['episode_id'] == added.episode_id for row in packet['items'])
    assert packet['coverage']['complete'] is False and await evidence.validate()
    await engine.forget('alpha', added.episode_id)
    assert not await evidence.validate()


@pytest.mark.parametrize('changes', [{'space':'other'}, {'metadata':{'team':'red'}},
                                    {'metadata':{'team':'blue','session_id':'current'}}, {'source':'current'}])
async def test_read_cannot_expand_foreign_or_excluded_sources(engine, changes):
    args = {'space':'alpha', 'metadata':{'team':'blue'}, **changes}
    _, chunks = await document(engine, **args)
    result = await box(engine).run('read_memory', {'chunk_id':chunks[0].chunk_id})
    assert result['status'] == 'unavailable' and result['items'] == []


@pytest.mark.parametrize('args', [{'chunk_id':True}, {'chunk_id':0}, {'chunk_id':1,'before':5},
    {'chunk_id':1,'after':-1}, {'chunk_id':1,'where':{'team':'red'}}])
async def test_read_validates_before_store_io(engine, monkeypatch, args):
    read = AsyncMock(side_effect=AssertionError('invalid arguments reached storage'))
    monkeypatch.setattr(engine.documents, 'get_chunks', read)
    result = await box(engine).run('read_memory', args)
    assert result['error'] == 'invalid_arguments'
    read.assert_not_called()


async def test_unbounded_fallback_is_not_used(engine, monkeypatch):
    _, chunks = await document(engine, metadata={'team':'blue'})
    store = engine.documents

    class Legacy:
        async def get_chunks(self, *args): return await store.get_chunks(*args)
        async def get_episode(self, *args): return await store.get_episode(*args)
        async def revision(self, *args): return await store.revision(*args)
        async def chunks_of(self, *args): raise AssertionError('unbounded fallback')

    monkeypatch.setattr(engine, 'documents', Legacy())
    result = await box(engine).run('read_memory', {'chunk_id':chunks[0].chunk_id})
    assert result['error'] == 'unsupported_chunk_window'


async def test_oversized_read_is_refused_whole(engine):
    _, chunks = await document(engine, metadata={'team':'blue'})
    result = await box(engine, max_result_bytes=512).run('read_memory', {'chunk_id':chunks[3].chunk_id,'before':3,'after':3})
    assert result['status'] == 'unavailable' and result['items'] == []
    assert result['error'] == 'output_bytes'


async def test_malformed_episode_metadata_cannot_bypass_session_exclusion(engine, monkeypatch):
    _, chunks = await document(engine, metadata={'team':'blue'})
    original = engine.documents.get_episode

    async def malformed(*args):
        episode = await original(*args)
        return episode.model_copy(update={'metadata':{'team':'blue', 'session_id':['current']}})

    monkeypatch.setattr(engine.documents, 'get_episode', malformed)
    evidence = await box(engine).prepare('read_memory', {'chunk_id':chunks[0].chunk_id})
    packet = json.loads(evidence.payload)
    assert packet['status'] == 'unavailable' and packet['items'] == []
    assert not evidence.evidence_ids


async def test_shared_evidence_capture_rejects_malformed_episode_metadata(engine, monkeypatch):
    from scone_memory.agents.tool_evidence import prepare_tool_evidence

    _, chunks = await document(engine, metadata={'team':'blue'})
    packet = await box(engine).run('read_memory', {'chunk_id':chunks[0].chunk_id})
    assert packet['status'] == 'prepared'
    original = engine.documents.get_episode

    async def malformed(*args):
        episode = await original(*args)
        return episode.model_copy(update={'metadata':{'team':'blue', 'session_id':['current']}})

    monkeypatch.setattr(engine.documents, 'get_episode', malformed)
    with pytest.raises(ValueError):
        await prepare_tool_evidence(engine, 'alpha', RecallScope.validated(where={'team':'blue'}),
                                   'current', packet, timeout_s=2.0)


@pytest.mark.parametrize('change', ['foreign', 'span', 'duplicate', 'oversize', 'deleted'])
async def test_corrupt_or_changed_window_is_not_returned(engine, monkeypatch, change):
    added, chunks = await document(engine, metadata={'team':'blue'})
    original = engine.documents.page_chunks

    async def bad_page(*args, **kwargs):
        rows = await original(*args, **kwargs)
        if change == 'foreign':
            rows[0] = rows[0].model_copy(update={'space':'other'})
        elif change == 'span':
            rows[0] = rows[0].model_copy(update={'text':'INVENTED_NEIGHBOR'})
        elif change == 'duplicate':
            rows[1] = rows[0]
        elif change == 'oversize':
            rows *= 20
        else:
            await engine.forget('alpha', added.episode_id)
        return rows

    monkeypatch.setattr(engine.documents, 'page_chunks', bad_page)
    result = await box(engine).run('read_memory', {'chunk_id':chunks[4].chunk_id})
    assert result['status'] == 'unavailable' and result['items'] == []
    assert 'INVENTED_NEIGHBOR' not in json.dumps(result)
