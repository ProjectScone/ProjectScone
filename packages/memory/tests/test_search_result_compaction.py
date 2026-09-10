"""Repeated search presentation can be compacted without caching retrieval."""
import asyncio
import json

import httpx
import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolStep
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.providers.tool_chat import SelfHostedToolChat
from test_evidence_tool_loop import Script, binding, search


@pytest.mark.parametrize('enabled', [False, True])
async def test_identical_searches_execute_but_optional_compaction_references_original(engine, monkeypatch, enabled):
    await engine.remember('alpha', 'Juniper uses Polaris for calibration. ' * 20, metadata={'team':'blue'})
    tools = binding(engine)
    original = tools.prepare
    requests = []
    async def prepare(name, arguments):
        requests.append((name, arguments))
        return await original(name, arguments)
    monkeypatch.setattr(tools, 'prepare', prepare)
    model = Script(search('first'), search('second'), search('third'), ToolStep(content='Polaris'))
    result = await EvidenceToolLoop(model, tools, compact_search_results=enabled).run(
        [{'role':'user', 'content':'Which calibration star?'}])
    packets = [json.loads(row['content']) for row in model.requests[-1][0] if row['role']=='tool']
    assert len(requests) == result.tool_calls == len(packets) == 3
    assert result.model_calls == 4 and result.text == 'Polaris' and await result.validate()
    assert len(result.evidence_packets) == 3
    assert len(set(result.evidence_packets)) == 1
    assert [outcome.reused for outcome in result.tool_outcomes] == [False, enabled, enabled]
    if enabled:
        assert all(packet['reuse'] == {'tool_result_number':1, 'new_evidence_count':0, 'searched_again':True}
                   for packet in packets[1:])
        assert all(outcome.output_bytes < result.tool_outcomes[0].output_bytes for outcome in result.tool_outcomes[1:])
    else:
        assert packets[0] == packets[1] == packets[2]


async def test_changed_search_results_are_offered_in_full(engine):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    await engine.remember('alpha', 'Juniper also tracks Sirius.', metadata={'team':'blue'})
    model = Script(search('first', limit=1), search('second', limit=2), ToolStep(content='Polaris and Sirius'))
    result = await EvidenceToolLoop(model, binding(engine), compact_search_results=True).run(
        [{'role':'user', 'content':'Which stars?'}])
    packets = [json.loads(row['content']) for row in model.requests[-1][0] if row['role']=='tool']
    assert [len(packet['items']) for packet in packets] == [1, 2]
    assert not any(outcome.reused for outcome in result.tool_outcomes)
    assert await result.validate()


async def test_identical_query_can_discover_newly_added_records(engine):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    model = Script()
    async def add_then_search():
        await engine.remember('alpha', 'Juniper now also uses Sirius.', metadata={'team':'blue'})
        return search('second')
    async def inspect_and_cancel():
        packet = json.loads(model.requests[-1][0][-1]['content'])
        assert 'reuse' not in packet
        assert any('Sirius' in item['text'] for item in packet['items'])
        raise asyncio.CancelledError()
    model.steps = [search('first'), add_then_search, inspect_and_cancel]
    with pytest.raises(asyncio.CancelledError):
        await EvidenceToolLoop(model, binding(engine), compact_search_results=True).run(
            [{'role':'user', 'content':'Which stars?'}])


async def test_empty_searches_are_not_represented_as_retained_evidence(engine):
    model = Script(search('first'), search('second'), ToolStep(content='No evidence returned'))
    result = await EvidenceToolLoop(model, binding(engine), compact_search_results=True).run(
        [{'role':'user', 'content':'Which stars?'}])
    assert result.source_status == 'none' and not result.evidence_packets
    assert not any(outcome.reused for outcome in result.tool_outcomes)


async def test_deletion_after_compaction_still_withholds_the_answer(engine):
    source = await engine.remember('alpha', 'Juniper uses Polaris. ' * 20, metadata={'team':'blue'})
    async def forget_then_answer():
        await engine.forget('alpha', source.episode_id)
        return ToolStep(content='Polaris')
    model = Script(search('first'), search('second'), forget_then_answer)
    with pytest.raises(RuntimeError, match='evidence changed'):
        await EvidenceToolLoop(model, binding(engine), compact_search_results=True).run(
            [{'role':'user', 'content':'Which star?'}])
    assert json.loads(model.requests[-1][0][-1]['content'])['reuse']['searched_again'] is True


@pytest.mark.parametrize('value', ['true', 1, None])
def test_compaction_configuration_is_strict(engine, value):
    with pytest.raises(ValueError, match='compact_search_results'):
        EvidenceToolLoop(Script(), binding(engine), compact_search_results=value)


async def test_same_ids_with_changed_ranking_are_not_compacted(engine):
    await engine.remember('alpha', 'Juniper Polaris calibration observatory.', metadata={'team':'blue'})
    await engine.remember('alpha', 'Juniper Sirius telescope survey.', metadata={'team':'blue'})
    model = Script(search('first', query='Polaris calibration'), search('second', query='Sirius telescope'),
        ToolStep(content='Two sources'))
    result = await EvidenceToolLoop(model, binding(engine), compact_search_results=True).run(
        [{'role':'user', 'content':'Which stars?'}])
    packets = [json.loads(row['content']) for row in model.requests[-1][0] if row['role']=='tool']
    orders = [[item['chunk_id'] for item in packet['items']] for packet in packets]
    assert set(orders[0]) == set(orders[1]) and orders[0] != orders[1]
    assert not any(outcome.reused for outcome in result.tool_outcomes)


@pytest.mark.parametrize('structured', [False, True])
async def test_compacted_initial_search_keeps_provider_pairing_and_final_evidence(engine, structured):
    text = 'Juniper uses Polaris for calibration. ' * 20
    await engine.remember('alpha', text, metadata={'team':'blue'})
    requests = []
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 3:
            message = {'role':'assistant', 'content':'Polaris'}
        elif structured:
            message = {'role':'assistant', 'content':'{"action":"search_memory","query":"Juniper","limit":5}'}
        else:
            message = {'role':'assistant', 'content':None, 'tool_calls':[{
                'id':f'search-{len(requests)}', 'type':'function',
                'function':{'name':'search_memory', 'arguments':'{"query":"Juniper","limit":5}'}}]}
        return httpx.Response(200, json={'choices':[{
            'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop', 'message':message}]})
    provider = SelfHostedStructuredToolChat if structured else SelfHostedToolChat
    model = provider('http://127.0.0.1:11434/v1', 'test-model', transport=httpx.MockTransport(serve))
    result = await EvidenceToolLoop(model, binding(engine), initial_search=True, compact_search_results=True,
        limits=ToolLoopLimits(max_tool_calls=3)).run([{'role':'user', 'content':'Juniper'}])
    assert result.text == 'Polaris' and result.tool_calls == result.model_calls == 3
    assert result.tool_outcomes[0].origin == 'host'
    assert [outcome.reused for outcome in result.tool_outcomes] == [False, True, True]
    assert await result.validate() and result.verified_accuracy is False
    assert 'searched_again' in json.dumps(requests[1]['messages'])
    assert 'Juniper uses Polaris for calibration.' in json.dumps(requests[-1]['messages'])
    assert not requests[-1].get('tools')
