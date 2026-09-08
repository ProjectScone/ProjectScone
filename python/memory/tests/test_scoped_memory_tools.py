import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from scone_memory.core.ports import NewFact
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


async def seed(engine, subject, obj, *, team='blue', space='alpha', kind='file', source='manuals/doc',
               created_at='2025-01-01T00:00:00Z', session_id=None):
    quote=f'{subject} depends on {obj}.'
    metadata={'team':team}
    if session_id: metadata['session_id']=session_id
    episode=await engine.remember(space, quote, kind=kind, source=source, created_at=created_at, metadata=metadata)
    fact=await engine.documents.insert_fact(NewFact(space=space,subject=subject,predicate='depends on',object=obj,
        valid_from='2025-01-01T00:00:00Z',source_episode_id=episode.episode_id,quote=quote))
    return fact


def box(engine, **kwargs):
    return ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated(where={'team':'blue'},
        kind='file', source_prefix='manuals/', since='2024-01-01T00:00:00Z', until='2025-12-31T00:00:00Z'),
        exclude_session_id='current', **kwargs)


async def test_search_respects_all_host_constraints_and_has_quoted_provenance(engine):
    first=await seed(engine,'aster','beacon')
    for changes in ({'team':'red'}, {'space':'other'}, {'kind':'note'}, {'source':'private/doc'},
                    {'created_at':'2023-01-01T00:00:00Z'}, {'created_at':'2026-01-01T00:00:00Z'}, {'session_id':'current'}):
        await seed(engine,'aster','PRIVATE_MARKER',**changes)
    result=await box(engine).run('search_memory', {'query':'aster'})
    assert result['ok'] is True
    assert result['facts'] and result['items']
    assert {f['fact_id'] for f in result['facts']} == {first.fact_id}
    assert result['facts'][0]['quote']==first.quote
    assert result['facts'][0]['source_episode_id']==first.source_episode_id
    assert result['verified_accuracy'] is False
    assert 'PRIVATE_MARKER' not in json.dumps(result)


async def test_trace_restricts_every_expanded_source_not_only_seed(engine):
    first=await seed(engine,'aster','beacon')
    bridge=await seed(engine,'beacon','cedar')
    await seed(engine,'cedar','PRIVATE_MARKER',team='red')
    result=await box(engine).run('trace_memory', {'seed_fact_id':first.fact_id})
    assert result['ok'] is True
    assert 'PRIVATE_MARKER' not in json.dumps(result)
    assert {f['fact_id'] for f in result['claims']} == {first.fact_id,bridge.fact_id}


@pytest.mark.parametrize('name,args', [('add_memory',{'content':'WRITE_MARKER'}),('read_profile',{}),
    ('search_memory',{'query':'aster','where':{'team':'red'}}),('trace_memory',{'seed_fact_id':1,'space':'other'})])
async def test_no_write_or_scope_override_can_reach_engine(engine, monkeypatch, name, args):
    read=AsyncMock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(engine,'recall',read)
    revision=await engine.documents.revision('alpha')
    result=await box(engine).run(name,args)
    assert result['ok'] is False
    read.assert_not_called()
    assert await engine.documents.revision('alpha')==revision


def test_schemas_expose_only_supported_read_operations():
    tools=box(None)
    assert [r['function']['name'] for r in tools.openai()] == ['search_memory','trace_memory']
    assert [r['name'] for r in tools.anthropic()] == ['search_memory','trace_memory']
    assert all('space' not in r['function']['parameters']['properties'] for r in tools.openai())


async def test_mutating_original_scope_cannot_change_later_calls(engine):
    await seed(engine,'aster','beacon')
    await seed(engine,'aster','PRIVATE_MARKER',team='red')
    scope=RecallScope.validated(where={'team':'blue'})
    tools=ScopedMemoryTools(engine,'alpha',scope=scope)
    object.__setattr__(scope,'where',(('team','red'),))
    result=await tools.run('search_memory',{'query':'aster'})
    assert 'PRIVATE_MARKER' not in json.dumps(result)


async def test_provider_error_text_is_not_returned(engine, monkeypatch):
    monkeypatch.setattr(engine,'recall',AsyncMock(side_effect=RuntimeError('PRIVATE_ENDPOINT_SECRET')))
    result=await box(engine).run('search_memory',{'query':'aster'})
    assert result['ok'] is False
    assert 'PRIVATE_ENDPOINT_SECRET' not in json.dumps(result)


async def test_cancellation_propagates(engine, monkeypatch):
    entered=asyncio.Event()
    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(engine,'recall',wait)
    task=asyncio.create_task(box(engine).run('search_memory',{'query':'aster'}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task


@pytest.mark.parametrize('args', [{}, {'query':True}, {'query':' '}, {'query':'x','limit':True}, {'query':'x','limit':21}])
async def test_invalid_search_arguments_fail_without_read(args):
    result=await box(None).run('search_memory',args)
    assert result['ok'] is False and result['error']=='invalid_arguments'


@pytest.mark.parametrize('changes', [{'team':'red'}, {'kind':'note'}, {'source':'private/doc'},
    {'created_at':'2023-01-01T00:00:00Z'}, {'session_id':'current'}])
async def test_trace_never_exposes_an_ineligible_seed(engine, changes):
    first=await seed(engine,'PRIVATE_MARKER','hidden',**changes)
    result=await box(engine).run('trace_memory',{'seed_fact_id':first.fact_id})
    assert result['ok'] is True and result['status']=='empty'
    assert result['claims']==[]
    assert 'PRIVATE_MARKER' not in json.dumps(result)


async def test_whole_result_budget_rejects_instead_of_slicing_quotes(engine):
    await seed(engine,'aster','z'*2000)
    result=await box(engine,max_result_bytes=512).run('search_memory',{'query':'aster'})
    assert result['ok'] is False and result['error']=='output_bytes'
    assert result['facts']==result['items']==[]
    assert 'zzzz' not in json.dumps(result)


async def test_combined_result_limit_is_enforced_and_omission_visible(engine):
    for i in range(5): await seed(engine,'aster'+str(i),'beacon')
    result=await box(engine).run('search_memory',{'query':'beacon','limit':1})
    assert result['ok'] is True
    assert len(result['items'])+len(result['facts'])==1
    assert result['coverage']['truncated'] is True
    assert result['coverage']['output_omitted_count']>0


async def test_slow_search_is_explicit_timeout_and_does_not_complete_late(engine, monkeypatch):
    entered=asyncio.Event()
    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(engine,'recall',wait)
    result=await box(engine,timeout_s=1).run('search_memory',{'query':'aster'})
    assert entered.is_set() and result['ok'] is False
    assert result['error']=='timeout'


async def test_swallowed_cancel_cannot_return_success_after_deadline(monkeypatch):
    tools=box(None,timeout_s=1)
    async def ignore_cancel(args):
        try: await asyncio.Event().wait()
        except asyncio.CancelledError:
            asyncio.current_task().uncancel()
            return {'ok':True, 'text':'LATE_MARKER'}
    monkeypatch.setattr(tools,'_search',ignore_cancel)
    result=await tools.run('search_memory',{'query':'aster'})
    assert result['ok'] is False and result['error']=='timeout'
    assert 'LATE_MARKER' not in json.dumps(result)


@pytest.mark.parametrize('options', [{'timeout_s':True}, {'timeout_s':float('inf')}, {'timeout_s':31},
    {'max_result_bytes':True}, {'max_result_bytes':511}, {'max_result_bytes':64001}, {'exclude_session_id':'bad/session'}])
def test_host_configuration_is_validated_before_access(options):
    with pytest.raises(ValueError):
        ScopedMemoryTools(None,'alpha',scope=RecallScope.validated(),**options)


def test_forged_scope_is_validated_before_access():
    scope=RecallScope.validated()
    object.__setattr__(scope,'kind','not-a-kind')
    with pytest.raises(ValueError): ScopedMemoryTools(None,'alpha',scope=scope)


async def test_schema_mutation_cannot_expand_the_actual_tool_allowlist():
    tools=box(None)
    schema=tools.openai()
    schema[0]['function']['name']='add_memory'
    assert (await tools.run('add_memory',{'content':'WRITE_MARKER'}))['ok'] is False
    assert tools.openai()[0]['function']['name']=='search_memory'


async def test_backend_uncancel_does_not_clear_callers_external_cancellation(monkeypatch):
    tools=box(None)
    entered=asyncio.Event()
    async def ignore_cancel(args):
        entered.set()
        try: await asyncio.Event().wait()
        except asyncio.CancelledError:
            asyncio.current_task().uncancel()
            return {'ok':True,'text':'LATE_MARKER'}
    monkeypatch.setattr(tools,'_search',ignore_cancel)
    task=asyncio.create_task(tools.run('search_memory',{'query':'aster'}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task


async def test_scoped_search_drops_source_deleted_after_selection(engine, monkeypatch):
    from scone_memory.integrations.scoped_tools import _RetainCandidates
    first=await seed(engine,'aster','beacon')
    original=_RetainCandidates.assess
    async def select_then_delete(self, question, candidates):
        decision=await original(self,question,candidates)
        await engine.forget('alpha',first.source_episode_id)
        return decision
    monkeypatch.setattr(_RetainCandidates,'assess',select_then_delete)
    result=await box(engine).run('search_memory',{'query':'aster'})
    assert result['ok'] is True and result['status']=='empty'
    assert result['facts']==result['items']==[]
    assert 'stale_evidence' in result['coverage']['reasons']


async def test_relationship_source_must_also_match_host_scope(engine):
    from scone_memory.core.ports import NewFactLink
    first=await seed(engine,'aster','beacon')
    second=await seed(engine,'cedar','denver')
    source=await engine.remember('alpha','PRIVATE_RELATION_MARKER',kind='file',source='manuals/link',
        metadata={'team':'red'},created_at='2025-01-01T00:00:00Z')
    await engine.documents.insert_fact_link(NewFactLink(space='alpha',from_fact=first.fact_id,to_fact=second.fact_id,
        kind='supports',created_at='2025-01-01T00:00:00Z',source_episode_id=source.episode_id,quote='PRIVATE_RELATION_MARKER'))
    result=await box(engine).run('trace_memory',{'seed_fact_id':first.fact_id})
    assert result['ok'] is True
    assert result['relations']==[]
    assert {row['fact_id'] for row in result['claims']}=={first.fact_id}
    assert 'PRIVATE_RELATION_MARKER' not in json.dumps(result)
