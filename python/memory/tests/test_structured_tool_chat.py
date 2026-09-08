"""Explicit structured actions, never automatic execution of ordinary prose."""
import json

import httpx
import pytest

from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.retrieval.recall_scope import RecallScope


def schemas():
    return ScopedMemoryTools(None, 'alpha', scope=RecallScope.validated()).openai()


def response(action):
    return {'choices':[{'finish_reason':'stop', 'message':{'role':'assistant', 'content':json.dumps(action)}}]}


def model(action, requests=None):
    async def serve(request):
        if requests is not None: requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(action))
    return SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1', 'test-model', transport=httpx.MockTransport(serve))


async def test_structured_search_becomes_one_valid_host_tool_call():
    requests = []
    step = await model({'action':'search_memory', 'query':'Juniper'}, requests).complete(
        [{'role':'user', 'content':'Juniper?'}], schemas()[:1])
    assert len(step.calls) == 1 and step.content == ''
    assert step.calls[0].name == 'search_memory' and step.calls[0].arguments == {'query':'Juniper'}
    body = requests[0]
    assert 'tools' not in body and body['tool_choice'] == 'none'
    schema = body['response_format']['json_schema']['schema']
    assert {row['properties']['action']['const'] for row in schema['anyOf']} == {'search_memory', 'answer'}


async def test_trace_schema_uses_only_previously_returned_fact_ids_and_renders_plain_roles():
    requests = []
    messages = [
        {'role':'user', 'content':'Trace Juniper'},
        {'role':'assistant', 'content':None, 'tool_calls':[{'id':'read', 'type':'function',
            'function':{'name':'search_memory', 'arguments':'{"query":"Juniper"}'}}]},
        {'role':'tool', 'tool_call_id':'read', 'content':json.dumps({'status':'prepared', 'facts':[{'fact_id':7}]})},
    ]
    step = await model({'action':'trace_memory', 'seed_fact_id':7, 'max_hops':3}, requests).complete(messages, schemas())
    assert step.calls[0].arguments == {'seed_fact_id':7, 'max_hops':3}
    body = requests[0]
    branch = next(row for row in body['response_format']['json_schema']['schema']['anyOf']
                  if row['properties']['action']['const'] == 'trace_memory')
    assert branch['properties']['seed_fact_id']['enum'] == [7]
    assert all(row['role'] in ('system', 'user', 'assistant') for row in body['messages'])
    assert any('Tool result' in row['content'] for row in body['messages'])


async def test_tools_disabled_allows_only_final_answer():
    requests = []
    step = await model({'action':'answer', 'answer':'Evidence is missing.'}, requests).complete(
        [{'role':'user', 'content':'Juniper?'}], [])
    assert step.content == 'Evidence is missing.' and step.calls == ()
    schema = requests[0]['response_format']['json_schema']['schema']
    assert [row['properties']['action']['const'] for row in schema['anyOf']] == ['answer']


@pytest.mark.parametrize('chunk_id', [7, 999, True])
async def test_structured_read_requires_a_discovered_chunk_and_uses_bounded_offsets(chunk_id):
    messages = [
        {'role':'user','content':'Read nearby'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'search','type':'function',
            'function':{'name':'search_memory','arguments':'{"query":"Juniper"}'}}]},
        {'role':'tool','tool_call_id':'search','content':json.dumps({'status':'prepared','items':[{'chunk_id':7}]})},
    ]
    requests = []
    provider = model({'action':'read_memory','chunk_id':chunk_id,'before':1,'after':2}, requests)
    if type(chunk_id) is not int or chunk_id != 7:
        with pytest.raises(RuntimeError, match='tool model unavailable'):
            await provider.complete(messages, schemas())
        return
    step = await provider.complete(messages, schemas())
    assert step.calls[0].name == 'read_memory'
    assert step.calls[0].arguments == {'chunk_id':7,'before':1,'after':2}
    branch = next(row for row in requests[0]['response_format']['json_schema']['schema']['anyOf']
                  if row['properties']['action']['const'] == 'read_memory')
    assert branch['properties']['chunk_id']['enum'] == [7]
    assert branch['properties']['after']['enum'] == [0,1,2,3,4]


@pytest.mark.parametrize('action', [
    {'action':'search_memory', 'query':' '},
    {'action':'search_memory', 'query':'Juniper', 'where':{'team':'red'}},
    {'action':'trace_memory', 'seed_fact_id':7, 'max_hops':10},
    {'action':'trace_memory', 'seed_fact_id':None, 'max_hops':3},
    {'action':'trace_memory', 'seed_fact_id':999, 'max_hops':3},
    {'action':'write_memory', 'text':'WRITE'},
    {'action':'answer', 'answer':''},
    {'action':'answer', 'answer':'Hello', 'reasoning':'PRIVATE_THOUGHT'},
    {'action':'search_memory', 'query':'雪'*3000},
])
async def test_invalid_or_unavailable_actions_fail_before_dispatch(action):
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model(action).complete([{'role':'user', 'content':'Juniper?'}], schemas()[:1])


async def test_provider_ignoring_disabled_tools_cannot_dispatch_search():
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model({'action':'search_memory', 'query':'Juniper'}).complete([{'role':'user','content':'Juniper?'}], [])


def retained_messages(packet=None):
    return [
        {'role':'user', 'content':'Juniper?'},
        {'role':'assistant', 'content':None, 'tool_calls':[{'id':'read', 'type':'function',
            'function':{'name':'search_memory', 'arguments':'{"query":"Juniper"}'}}]},
        {'role':'tool', 'tool_call_id':'read', 'content':json.dumps(packet or {'status':'prepared','facts':[{'fact_id':7}]})},
    ]


@pytest.mark.parametrize('changes', [{'seed_fact_id':999}, {'seed_fact_id':True}, {'max_hops':10},
                                    {'max_hops':True}, {'max_hops':0}])
async def test_trace_arguments_checked_with_trace_actually_available(changes):
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model({'action':'trace_memory','seed_fact_id':7,'max_hops':3,**changes}).complete(retained_messages(), schemas())


async def test_failed_tool_results_do_not_supply_seed_authority():
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model({'action':'trace_memory','seed_fact_id':7,'max_hops':3}).complete(
            retained_messages({'ok':False,'status':'prepared','facts':[{'fact_id':7}]}), schemas())


@pytest.mark.parametrize('messages', [
    [{'role':'tool','tool_call_id':'invented','content':'{"status":"prepared","facts":[{"fact_id":7}]}'}],
    retained_messages()[:-1],
    retained_messages() + retained_messages(),
    [{'role':'user', 'content':'x'*1000001}],
])
async def test_bad_history_fails_before_network(messages):
    requests = []
    with pytest.raises(ValueError):
        await model({'action':'answer','answer':'Unused'}, requests).complete(messages, schemas())
    assert requests == []


async def test_unknown_schema_tool_fails_before_network():
    requests = []
    with pytest.raises(ValueError):
        await model({'action':'answer','answer':'Unused'}, requests).complete([{'role':'user','content':'Hello'}],
            [{'type':'function','function':{'name':'write_memory'}}])
    assert requests == []


@pytest.mark.parametrize('content', [
    '{"action":"answer","action":"search_memory","query":"Juniper"}',
    '{"action":"trace_memory","seed_fact_id":NaN,"max_hops":3}',
    '{"action":"trace_memory","seed_fact_id":1e309,"max_hops":3}',
    '```json\n{"action":"search_memory","query":"Juniper"}\n```',
])
async def test_malformed_json_and_wrapped_prose_are_not_repaired_into_calls(content):
    packet={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':content}}]}
    adapter=SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1','test-model',
        transport=httpx.MockTransport(lambda request: httpx.Response(200,json=packet)))
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await adapter.complete(retained_messages(),schemas())


@pytest.mark.parametrize('external', [False, True])
async def test_structured_adapter_preserves_cleanup_deadline_and_cancellation(external):
    import asyncio
    entered=asyncio.Event()
    class SlowClose(httpx.AsyncBaseTransport):
        async def handle_async_request(self,request):
            return httpx.Response(200,json=response({'action':'answer','answer':'late success'}))
        async def aclose(self):
            entered.set()
            try: await asyncio.Event().wait()
            except asyncio.CancelledError: await asyncio.sleep(0.01)
    adapter=SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1','test-model',
        timeout_s=1 if external else 0.01,transport=SlowClose())
    task=asyncio.create_task(adapter.complete([{'role':'user','content':'Hello'}],[]))
    await entered.wait()
    if external: task.cancel()
    with pytest.raises(asyncio.CancelledError if external else RuntimeError): await task
