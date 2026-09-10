"""Behavioral coverage for model-requested, host-scoped memory reads."""
import asyncio
import json

import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolStep, ToolCall
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


class Script:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    async def complete(self, messages, tools):
        self.requests.append((messages, tools))
        step = self.steps.pop(0)
        if callable(step):
            return await step()
        return step


def search(call_id='read-1', **arguments):
    return ToolStep(calls=(ToolCall(id=call_id, name='search_memory', arguments={'query': 'Juniper', **arguments}),))


def binding(engine):
    return ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated(where={'team': 'blue'}),
                             exclude_session_id='current')


async def test_read_tool_is_only_available_after_search_returns_a_chunk(engine):
    from test_chunk_window import document
    await document(engine, metadata={'team':'blue'})
    model = Script()

    async def read_discovered():
        messages, tools = model.requests[-1]
        packet = json.loads(messages[-1]['content'])
        assert 'read_memory' in [row['function']['name'] for row in tools]
        return ToolStep(calls=(ToolCall(id='neighbors', name='read_memory',
            arguments={'chunk_id':packet['items'][0]['chunk_id'], 'before':1, 'after':2}),))

    model.steps = [search(query='Section 4 calibration', limit=1), read_discovered, ToolStep(content='Checked nearby evidence.')]
    result = await EvidenceToolLoop(model, binding(engine)).run([{'role':'user','content':'What does Section 4 say?'}])
    assert [row['function']['name'] for row in model.requests[0][1]] == ['search_memory']
    packet = json.loads(result.evidence_packets[-1])
    assert packet['coverage']['mode'] == 'chunk_window' and len(packet['items']) > 1
    assert result.tool_outcomes[-1].name == 'read_memory' and await result.validate()


async def test_read_cannot_probe_an_undiscovered_chunk_even_inside_scope(engine, monkeypatch):
    from test_chunk_window import document
    _, chunks = await document(engine, metadata={'team':'blue'})
    tools = binding(engine)
    async def forbidden(*args):
        pytest.fail('undiscovered chunk reached storage')
    monkeypatch.setattr(tools, 'prepare', forbidden)
    model = Script(ToolStep(calls=(ToolCall(id='probe', name='read_memory', arguments={'chunk_id':chunks[0].chunk_id}),)),
                   ToolStep(content='Search first.'))
    result = await EvidenceToolLoop(model, tools).run([{'role':'user','content':'Read memory'}])
    assert result.tool_outcomes[0].error == 'search_for_chunk_first'
    assert result.source_status == 'none'


async def test_search_is_paired_and_final_answer_has_retained_sources(engine):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    await engine.remember('alpha', 'Juniper PRIVATE_MARKER', metadata={'team': 'red'})
    model = Script(search(), ToolStep(content='Juniper uses Polaris.'))
    result = await EvidenceToolLoop(model, binding(engine)).run([{'role': 'user', 'content': 'What does Juniper use?'}])
    assert result.text == 'Juniper uses Polaris.'
    assert result.tool_calls == 1 and result.model_calls == 2
    assert result.source_status == 'retained' and result.verified_accuracy is False
    assert result.evidence_ids
    assert result.evidence_packets and json.loads(result.evidence_packets[0])['items']
    messages = model.requests[1][0]
    assert messages[-2]['tool_calls'][0]['id'] == messages[-1]['tool_call_id'] == 'read-1'
    assert str(episode.episode_id) in messages[-1]['content']
    assert 'PRIVATE_MARKER' not in json.dumps(messages)


async def test_source_deleted_during_generation_prevents_public_answer(engine):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    async def delete_then_answer():
        await engine.documents.delete_episode('alpha', episode.episode_id)
        return ToolStep(content='Must never publish this.')
    with pytest.raises(RuntimeError, match='evidence changed'):
        await EvidenceToolLoop(Script(search(), delete_then_answer), binding(engine)).run(
            [{'role': 'user', 'content': 'Juniper?'}])


async def test_exhausted_budget_pairs_all_calls_then_forces_final_without_tools(engine):
    model = Script(ToolStep(calls=(search('a').calls[0], search('b').calls[0])), ToolStep(content='No evidence.'))
    result = await EvidenceToolLoop(model, binding(engine), limits=ToolLoopLimits(max_tool_calls=1)).run(
        [{'role': 'user', 'content': 'Juniper?'}])
    messages, tools = model.requests[1]
    assert tools == []
    assert [row['tool_call_id'] for row in messages if row['role'] == 'tool'] == ['a', 'b']
    assert json.loads(messages[-1]['content'])['error'] == 'tool_budget'
    assert result.tool_calls == 1


async def test_unknown_tool_and_scope_injection_are_not_executed(engine):
    model = Script(ToolStep(calls=(ToolCall(id='a', name='add_memory', arguments={'content': 'WRITE'}),
                                   search('b', where={'team': 'red'}).calls[0])), ToolStep(content='Unavailable.'))
    revision = await engine.documents.revision('alpha')
    await EvidenceToolLoop(model, binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])
    results = [json.loads(m['content']) for m in model.requests[1][0] if m['role'] == 'tool']
    assert [r['error'] for r in results] == ['unknown_tool', 'invalid_arguments']
    assert await engine.documents.revision('alpha') == revision


async def test_duplicate_call_ids_fail_before_any_tool_runs(engine, monkeypatch):
    tools = binding(engine)
    async def forbidden(*args):
        pytest.fail('duplicate batch must not execute')
    monkeypatch.setattr(tools, 'prepare', forbidden)
    model = Script(ToolStep(calls=(search('a').calls[0], search('a').calls[0])))
    with pytest.raises(RuntimeError, match='tool protocol'):
        await EvidenceToolLoop(model, tools).run([{'role': 'user', 'content': 'Juniper?'}])


async def test_call_after_tools_disabled_fails(engine):
    model = Script(search('a'), search('b'))
    with pytest.raises(RuntimeError, match='tool protocol'):
        await EvidenceToolLoop(model, binding(engine), limits=ToolLoopLimits(max_tool_calls=1)).run(
            [{'role': 'user', 'content': 'Juniper?'}])


async def test_external_cancellation_survives_provider_suppression(engine):
    entered = asyncio.Event()
    async def suppress():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return ToolStep(content='late reply')
    task = asyncio.create_task(EvidenceToolLoop(Script(suppress), binding(engine)).run(
        [{'role': 'user', 'content': 'Juniper?'}]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_model_failure_is_sanitized(engine):
    async def fail():
        raise ValueError('PRIVATE_PROVIDER_KEY')
    with pytest.raises(RuntimeError, match='tool model unavailable') as error:
        await EvidenceToolLoop(Script(fail), binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])
    assert 'PRIVATE_PROVIDER_KEY' not in str(error.value)


async def test_deadline_survives_provider_suppression(engine):
    async def suppress():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return ToolStep(content='late reply')
    with pytest.raises(TimeoutError):
        await EvidenceToolLoop(Script(suppress), binding(engine), limits=ToolLoopLimits(timeout_s=0.01)).run(
            [{'role': 'user', 'content': 'Juniper?'}])


async def test_tool_round_limit_forces_one_final_request(engine):
    model = Script(search(), ToolStep(content='No evidence.'))
    await EvidenceToolLoop(model, binding(engine), limits=ToolLoopLimits(max_tool_rounds=1)).run(
        [{'role': 'user', 'content': 'Juniper?'}])
    assert len(model.requests) == 2 and model.requests[1][1] == []


async def test_oversize_initial_request_is_rejected_before_model(engine):
    model = Script(ToolStep(content='unused'))
    with pytest.raises(RuntimeError, match='transcript byte limit'):
        await EvidenceToolLoop(model, binding(engine), limits=ToolLoopLimits(max_transcript_bytes=1024)).run(
            [{'role': 'user', 'content': 'x' * 1024}])
    assert model.requests == []


async def test_oversize_reply_is_not_published(engine):
    with pytest.raises(RuntimeError, match='reply byte limit'):
        await EvidenceToolLoop(Script(ToolStep(content='雪' * 10)), binding(engine),
                              limits=ToolLoopLimits(max_reply_bytes=20)).run([{'role': 'user', 'content': 'Juniper?'}])


async def test_trace_snapshot_rejects_fact_changed_without_revision(engine):
    from scone_memory.core.ports import NewFact
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    fact = await engine.documents.insert_fact(NewFact(space='alpha', subject='Juniper', predicate='uses',
        object='Polaris', valid_from='2025-01-01T00:00:00Z', source_episode_id=episode.episode_id,
        quote='Juniper uses Polaris.'))
    evidence = await binding(engine).prepare('trace_memory', {'seed_fact_id': fact.fact_id})
    assert evidence.evidence_ids == (f'fact:{fact.fact_id}',)
    assert await evidence.validate()
    await engine.documents.update_fact(fact.model_copy(update={'object': 'Other'}))
    assert not await evidence.validate()


async def test_search_then_trace_exposes_directed_path(engine):
    from scone_memory.core.ports import NewFact
    facts = []
    for subject, obj in [('Juniper', 'Polaris'), ('Polaris', 'Beacon')]:
        quote = f'{subject} uses {obj}.'
        episode = await engine.remember('alpha', quote, metadata={'team': 'blue'})
        facts.append(await engine.documents.insert_fact(NewFact(space='alpha', subject=subject, predicate='uses',
            object=obj, valid_from='2025-01-01T00:00:00Z', source_episode_id=episode.episode_id, quote=quote)))
    model = Script(search(), ToolStep(calls=(ToolCall(id='trace', name='trace_memory',
        arguments={'seed_fact_id': facts[0].fact_id}),)), ToolStep(content='Juniper → Polaris → Beacon.'))
    result = await EvidenceToolLoop(model, binding(engine)).run([{'role': 'user', 'content': 'Trace Juniper.'}])
    packet = json.loads(model.requests[-1][0][-1]['content'])
    from scone_memory.retrieval.multihop import BoundedSubjectFacts
    if isinstance(engine.documents, BoundedSubjectFacts):
        assert packet['paths'] and len(packet['claims']) == 2
    else:
        assert not packet['paths'] and len(packet['claims']) == 1
        assert packet['coverage']['complete'] is False
        assert 'unsupported_bounded_subjects' in packet['coverage']['reasons']
    assert result.source_status == 'retained' and result.tool_calls == 2


async def test_source_change_during_prepare_returns_no_evidence(engine, monkeypatch):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    tools = binding(engine)
    original = tools.run
    async def changed(name, arguments):
        result = await original(name, arguments)
        await engine.documents.delete_episode('alpha', episode.episode_id)
        return result
    monkeypatch.setattr(tools, 'run', changed)
    result = await tools.prepare('search_memory', {'query': 'Juniper'})
    assert json.loads(result.payload)['error'] == 'evidence_unavailable'
    assert result.evidence_ids == ()


async def test_denials_count_toward_aggregate_tool_byte_budget(engine):
    model = Script(ToolStep(calls=tuple(search(str(i)).calls[0] for i in range(4))), ToolStep(content='unused'))
    with pytest.raises(RuntimeError, match='tool output byte limit'):
        await EvidenceToolLoop(model, binding(engine), limits=ToolLoopLimits(max_tool_bytes=512)).run(
            [{'role': 'user', 'content': 'Juniper?'}])
    assert len(model.requests) == 1


async def test_prepare_shares_deadline_between_search_and_snapshot(engine, monkeypatch):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    tools = binding(engine)
    packet = await tools.run('search_memory', {'query': 'Juniper'})
    assert packet['status'] == 'prepared'
    tools._timeout = 0.05
    original_get = engine.documents.get_episode
    async def slow_search(name, arguments):
        await asyncio.sleep(0.03)
        return packet
    async def slow_source(*args):
        await asyncio.sleep(0.03)
        return await original_get(*args)
    # Delay only snapshot reads, after the search has completed.
    async def delayed(name, arguments):
        result = await slow_search(name, arguments)
        monkeypatch.setattr(engine.documents, 'get_episode', slow_source)
        return result
    monkeypatch.setattr(tools, 'run', delayed)
    result = await tools.prepare('search_memory', {'query': 'Juniper'})
    assert json.loads(result.payload)['error'] == 'evidence_unavailable'
    assert result.evidence_ids == ()


async def test_native_http_adapter_and_real_memory_loop(engine):
    import httpx
    from scone_memory.providers.tool_chat import SelfHostedToolChat
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    requests = []
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'read', 'type': 'function',
                'function': {'name': 'search_memory', 'arguments': '{"query":"Juniper"}'}}]}
            reason = 'tool_calls'
        else:
            assert 'Juniper uses Polaris.' in body['messages'][-1]['content']
            message, reason = {'role': 'assistant', 'content': 'Juniper uses Polaris.'}, 'stop'
        return httpx.Response(200, json={'choices': [{'finish_reason': reason, 'message': message}]})
    model = SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model', transport=httpx.MockTransport(serve))
    result = await EvidenceToolLoop(model, binding(engine)).run([{'role': 'user', 'content': 'What does Juniper use?'}])
    assert result.source_status == 'retained' and len(requests) == 2


async def test_trace_is_exposed_only_after_retained_fact_discovery(engine):
    from scone_memory.core.ports import NewFact
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    await engine.documents.insert_fact(NewFact(space='alpha', subject='Juniper', predicate='uses', object='Polaris',
        valid_from='2025-01-01T00:00:00Z', source_episode_id=episode.episode_id, quote='Juniper uses Polaris.'))
    model = Script(search(), ToolStep(content='Juniper uses Polaris.'))
    await EvidenceToolLoop(model, binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])
    assert [row['function']['name'] for row in model.requests[0][1]] == ['search_memory']
    assert [row['function']['name'] for row in model.requests[1][1]] == ['search_memory', 'trace_memory', 'read_memory']


async def test_fabricated_trace_seed_never_reaches_tools(engine, monkeypatch):
    tools = binding(engine)
    async def forbidden(*args):
        pytest.fail('undiscovered seed must not execute')
    monkeypatch.setattr(tools, 'prepare', forbidden)
    model = Script(ToolStep(calls=(ToolCall(id='trace', name='trace_memory', arguments={'seed_fact_id': 999}),)),
                   ToolStep(content='Unavailable.'))
    result = await EvidenceToolLoop(model, tools).run([{'role': 'user', 'content': 'Juniper?'}])
    assert json.loads(model.requests[1][0][-1]['content'])['error'] == 'search_for_seed_first'
    assert result.source_status == 'none'
