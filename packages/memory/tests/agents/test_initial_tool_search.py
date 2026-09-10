"""Host-owned initial retrieval cannot be skipped by the answer model."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolCall, ToolLoopLimits, ToolStep
from ..agents.test_evidence_tool_loop import Script, binding


async def test_initial_search_precedes_model_and_retains_scoped_sources(engine):
    added = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    await engine.remember('alpha', 'Juniper PRIVATE_MARKER', metadata={'team':'red'})
    model = Script(ToolStep(content='Juniper uses Polaris.'))
    tools = binding(engine)
    original = tools.prepare
    tools.prepare = AsyncMock(wraps=original)
    result = await EvidenceToolLoop(model, tools, initial_search=True).run([
        {'role':'user','content':'What does Juniper use?'}])
    tools.prepare.assert_awaited_once_with('search_memory', {'query':'What does Juniper use?', 'limit':5})
    messages, schemas = model.requests[0]
    assert messages[-1]['role'] == 'tool'
    assert messages[-2]['tool_calls'][0]['id'] == messages[-1]['tool_call_id']
    assert 'PRIVATE_MARKER' not in json.dumps(messages)
    assert 'read_memory' in [schema['function']['name'] for schema in schemas]
    assert result.model_calls == 1 and result.tool_calls == 1
    assert result.tool_outcomes[0].origin == 'host'
    assert result.source_status == 'retained' and await result.validate()
    await engine.forget('alpha', added.episode_id)
    assert not await result.validate()


async def test_host_search_consumes_call_budget_without_consuming_a_model_round(engine):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    model = Script(ToolStep(content='Juniper uses Polaris.'))
    result = await EvidenceToolLoop(model, binding(engine), initial_search=True,
        limits=ToolLoopLimits(max_tool_calls=1)).run([{'role':'user','content':'Juniper?'}])
    assert model.requests[0][1] == []
    assert result.tool_calls == 1 and result.model_calls == 1


async def test_model_can_follow_initial_search_with_a_neighbor_read(engine):
    from ..retrieval.test_chunk_window import document
    await document(engine, metadata={'team':'blue'})
    model = Script()

    async def read():
        packet = json.loads(model.requests[-1][0][-1]['content'])
        return ToolStep(calls=(ToolCall(id='neighbor', name='read_memory',
            arguments={'chunk_id':packet['items'][0]['chunk_id'], 'before':1, 'after':2}),))

    model.steps = [read, ToolStep(content='Read the neighboring sections.')]
    result = await EvidenceToolLoop(model, binding(engine), initial_search=True).run([
        {'role':'user','content':'What do the calibration sections say?'}])
    assert [row.origin for row in result.tool_outcomes] == ['host', 'model']
    assert [row.name for row in result.tool_outcomes] == ['search_memory', 'read_memory']


@pytest.mark.parametrize('query', ['', ' ', 'x'*8001, '雪'*2667])
async def test_invalid_initial_query_fails_before_store_or_model(engine, query):
    model, tools = Script(), binding(engine)
    tools.prepare = AsyncMock(side_effect=AssertionError('invalid query reached storage'))
    with pytest.raises(ValueError, match='initial memory query'):
        await EvidenceToolLoop(model, tools, initial_search=True).run([{'role':'user','content':query}])
    tools.prepare.assert_not_called()
    assert model.requests == []


async def test_unavailable_initial_search_prevents_unanchored_generation(engine, monkeypatch):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    model = Script()
    monkeypatch.setattr(engine.documents, 'revision', AsyncMock(side_effect=RuntimeError('PRIVATE_STORE_ERROR')))
    with pytest.raises(RuntimeError, match='initial memory retrieval unavailable') as error:
        await EvidenceToolLoop(model, binding(engine), initial_search=True).run([{'role':'user','content':'Juniper?'}])
    assert model.requests == [] and 'PRIVATE' not in str(error.value)


async def test_empty_initial_search_is_visible_to_model_without_claiming_absence(engine):
    model = Script(ToolStep(content='No relevant evidence was returned.'))
    result = await EvidenceToolLoop(model, binding(engine), initial_search=True).run([{'role':'user','content':'Juniper?'}])
    packet = json.loads(model.requests[0][0][-1]['content'])
    assert packet['status'] == 'empty'
    assert result.source_status == 'none' and result.verified_accuracy is False


@pytest.mark.parametrize('budget', ['output', 'transcript'])
async def test_initial_search_respects_byte_budgets_before_model(engine, budget):
    await engine.remember('alpha', 'Juniper calibration uses Polaris. '*20, metadata={'team':'blue'})
    model = Script()
    limits = ToolLoopLimits(max_tool_bytes=512) if budget == 'output' else ToolLoopLimits(max_transcript_bytes=1024)
    with pytest.raises(RuntimeError):
        await EvidenceToolLoop(model, binding(engine), initial_search=True, limits=limits).run([
            {'role':'user','content':'Juniper calibration?'}])
    assert model.requests == []


async def test_initial_search_timeout_suppressed_by_store_still_prevents_model(engine, monkeypatch):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    original = engine.documents.revision
    async def slow(*args):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await original(*args)
    monkeypatch.setattr(engine.documents, 'revision', slow)
    model = Script()
    with pytest.raises(TimeoutError):
        await EvidenceToolLoop(model, binding(engine), initial_search=True,
            limits=ToolLoopLimits(timeout_s=0.05)).run([{'role':'user','content':'Juniper?'}])
    assert model.requests == []


async def test_external_cancel_during_initial_retrieval_stops_before_model(engine, monkeypatch):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    original, entered = engine.documents.revision, asyncio.Event()

    async def slow(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await original(*args)

    monkeypatch.setattr(engine.documents, 'revision', slow)
    model = Script()
    task = asyncio.create_task(EvidenceToolLoop(model, binding(engine), initial_search=True).run([
        {'role':'user','content':'Juniper?'}]))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.requests == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_model_cannot_reuse_host_call_id(engine):
    tools = binding(engine)
    tools.prepare = AsyncMock(wraps=tools.prepare)
    model = Script(ToolStep(calls=(ToolCall(id='host-initial-search', name='search_memory',
        arguments={'query':'Juniper'}),)))
    with pytest.raises(RuntimeError, match='invalid tool protocol'):
        await EvidenceToolLoop(model, tools, initial_search=True).run([{'role':'user','content':'Juniper?'}])
    assert tools.prepare.await_count == 1
