import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


def box(engine, **kwargs):
    return ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated(where={'team':'blue'}),
        exclude_session_id='current',enable_computation=True,**kwargs)


async def source(engine, text='Value: 12', **changes):
    options = {'metadata':{'team':'blue'}, **changes}
    space = options.pop('space','alpha')
    added = await engine.remember(space,text,**options)
    return added, (await engine.documents.chunks_of(space,added.episode_id))[0]


def request(chunk, quote='12'):
    return {'operation':'sum','left':[{'chunk_id':chunk.chunk_id,'quote':quote}]}


async def test_computation_is_opt_in_and_default_refuses_calls(engine):
    tools = ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated())
    assert 'compute_memory' not in [row['name'] for row in tools.anthropic()]
    assert (await tools.run('compute_memory',{}))['error'] == 'unknown_tool'
    assert [row['function']['name'] for row in box(engine).openai()][-1] == 'compute_memory'


async def test_calculation_keeps_sources_and_revalidates_before_publication(engine):
    added, chunk = await source(engine)
    evidence = await box(engine).prepare('compute_memory',request(chunk))
    packet = json.loads(evidence.payload)
    assert packet['status'] == 'prepared' and packet['computation']['value'] == '12'
    assert packet['items'][0]['text'] == chunk.text
    assert packet['items'][0]['episode_id'] == added.episode_id
    assert evidence.evidence_ids == (f'chunk:{chunk.chunk_id}',)
    assert await evidence.validate()
    await engine.forget('alpha',added.episode_id)
    assert not await evidence.validate()


@pytest.mark.parametrize('changes', [{'space':'other'}, {'metadata':{'team':'red'}},
    {'metadata':{'team':'blue','session_id':'current'}}, {'source':'current'}])
async def test_calculation_does_not_expose_foreign_or_excluded_passages(engine,changes):
    _, chunk = await source(engine,**changes)
    result = await box(engine).run('compute_memory',request(chunk))
    assert result['status'] == 'unavailable' and result['items'] == []
    assert 'computation' not in result and 'Value:' not in json.dumps(result)


async def test_invalid_arguments_and_unknown_tools_do_not_read_storage(engine,monkeypatch):
    read = AsyncMock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(engine.documents,'get_chunks',read)
    result = await box(engine).run('compute_memory',{'operation':'sum','left':[{'chunk_id':True,'quote':'12'}]})
    assert result['error'] == 'invalid_arguments'
    read.assert_not_called()


async def test_computation_uses_point_reads_without_window_or_full_scan(engine,monkeypatch):
    _, chunk = await source(engine)
    store = engine.documents
    class Points:
        async def get_chunks(self,*args): return await store.get_chunks(*args)
        async def get_episode(self,*args): return await store.get_episode(*args)
        async def revision(self,*args): return await store.revision(*args)
    monkeypatch.setattr(engine,'documents',Points())
    assert (await box(engine).run('compute_memory',request(chunk)))['computation']['value'] == '12'


async def test_output_limit_rejects_whole_calculation(engine):
    _, chunk = await source(engine)
    result = await box(engine,max_result_bytes=512).run('compute_memory',request(chunk))
    assert result['error'] == 'output_bytes' and 'computation' not in result


async def test_missing_number_is_sanitized(engine):
    _, chunk = await source(engine)
    result = await box(engine).run('compute_memory',request(chunk,'SECRET_INVENTED_NUMBER'))
    assert result['error'] == 'evidence_unavailable'
    assert 'SECRET' not in json.dumps(result)


async def test_source_change_during_computation_discards_result(engine,monkeypatch):
    added, chunk = await source(engine)
    read = engine.documents.get_chunks
    calls = 0
    async def changed(*args):
        nonlocal calls
        rows = await read(*args)
        calls += 1
        if calls == 2: await engine.forget('alpha',added.episode_id)
        return rows
    monkeypatch.setattr(engine.documents,'get_chunks',changed)
    result = await box(engine).run('compute_memory',request(chunk))
    assert result['status'] == 'unavailable' and 'computation' not in result


async def test_cancellation_propagates_from_calculation(engine,monkeypatch):
    _, chunk = await source(engine)
    entered = asyncio.Event()
    async def wait(*args):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(engine.documents,'get_chunks',wait)
    task = asyncio.create_task(box(engine).run('compute_memory',request(chunk)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task


@pytest.mark.parametrize('value',[1,'true',None])
def test_computation_flag_requires_boolean(value):
    with pytest.raises(ValueError):
        ScopedMemoryTools(None,'alpha',scope=RecallScope.validated(),enable_computation=value)


async def test_same_physical_quote_in_overlapping_chunks_cannot_be_counted_twice(engine):
    from scone_memory.core.ports import NewChunk
    _, chunk = await source(engine,'Équipe: Alba, Bo.')
    duplicate = (await engine.documents.insert_chunks([NewChunk(episode_id=chunk.episode_id,space='alpha',
        ordinal=chunk.ordinal+1,start=chunk.start,end=chunk.end,text=chunk.text,created_at=chunk.created_at)]))[0]
    result = await box(engine).run('compute_memory',{'operation':'count','left':[
        {'chunk_id':chunk.chunk_id,'quote':'Alba'}, {'chunk_id':duplicate.chunk_id,'quote':'Alba'}]})
    assert result['error'] == 'invalid_computation' and result['items'] == []


async def test_nonoverlapping_quotes_in_overlapping_chunks_are_valid(engine):
    from scone_memory.core.ports import NewChunk
    _, chunk = await source(engine,'Équipe: Alba, Bo.')
    duplicate = (await engine.documents.insert_chunks([NewChunk(episode_id=chunk.episode_id,space='alpha',
        ordinal=chunk.ordinal+1,start=chunk.start,end=chunk.end,text=chunk.text,created_at=chunk.created_at)]))[0]
    result = await box(engine).run('compute_memory',{'operation':'count','left':[
        {'chunk_id':chunk.chunk_id,'quote':'Alba'}, {'chunk_id':duplicate.chunk_id,'quote':'Bo'}]})
    assert result['computation']['value'] == '2'


async def test_read_timeout_cannot_publish_a_late_computation(engine,monkeypatch):
    _, chunk = await source(engine)
    original = engine.documents.get_chunks
    async def swallowed(*args):
        try: await asyncio.Event().wait()
        except asyncio.CancelledError:
            asyncio.current_task().uncancel()
            return await original(*args)
    monkeypatch.setattr(engine.documents,'get_chunks',swallowed)
    result = await box(engine,timeout_s=1).run('compute_memory',request(chunk))
    assert result['error'] == 'timeout' and 'computation' not in result


async def test_backend_errors_are_sanitized(engine,monkeypatch):
    _, chunk = await source(engine)
    monkeypatch.setattr(engine.documents,'get_chunks',AsyncMock(side_effect=RuntimeError('PRIVATE_SECRET')))
    result = await box(engine).run('compute_memory',request(chunk))
    assert result['error'] == 'store_error' and 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('content,start,end,quote',[
    ('Value: 1234',0,9,'12'), ('Value: 1234',9,11,'34'),
    ('Value: -12',8,10,'12'), ('Value: 1.25',9,11,'25'),
    ('Value: 12.5',0,9,'12'), ('Value: v12',8,10,'12'),
])
async def test_numeric_token_must_be_whole_in_full_source_not_only_chunk(engine,content,start,end,quote):
    from scone_memory.core.ports import NewChunk
    _, original = await source(engine,content)
    chunk = (await engine.documents.insert_chunks([NewChunk(episode_id=original.episode_id,space='alpha',
        ordinal=original.ordinal+1,start=start,end=end,text=content[start:end],created_at=original.created_at)]))[0]
    evidence = await box(engine).prepare('compute_memory',request(chunk,quote))
    result = json.loads(evidence.payload)
    assert result['error'] == 'numeric_literal_required' and result['items'] == []
    assert not evidence.evidence_ids and 'computation' not in result


async def test_full_source_numeric_boundaries_keep_chunk_relative_unicode_offsets(engine):
    from scone_memory.core.ports import NewChunk
    content='Équipe summary. Value: 12. End.'
    _, original=await source(engine,content)
    text='Value: 12'
    start=len(content[:content.index(text)].encode())
    chunk=(await engine.documents.insert_chunks([NewChunk(episode_id=original.episode_id,space='alpha',
        ordinal=original.ordinal+1,start=start,end=start+len(text.encode()),text=text,created_at=original.created_at)]))[0]
    result=await box(engine).run('compute_memory',request(chunk))
    assert result['computation']['value']=='12'
    assert result['computation']['left'][0]['start_char']==7
