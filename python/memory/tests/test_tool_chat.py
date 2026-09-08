import json

import httpx
import pytest

from scone_memory.providers.tool_chat import SelfHostedToolChat


def response(message, reason='stop'):
    return {'choices': [{'finish_reason': reason, 'message': {'role': 'assistant', **message}}]}


def tool(arguments='{"query":"Juniper"}', call_id='call-1'):
    return {'id': call_id, 'type': 'function', 'function': {'name': 'search_memory', 'arguments': arguments}}


async def test_native_tool_call_then_tool_less_final_request():
    requests = []
    async def serve(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=response({'content': None, 'tool_calls': [tool()]}, 'tool_calls'))
        return httpx.Response(200, json=response({'content': 'Juniper uses Polaris.', 'reasoning_content': 'PRIVATE_REASONING'}))
    model = SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model', transport=httpx.MockTransport(serve))
    step = await model.complete([{'role': 'user', 'content': 'Juniper?'}], [{'type': 'function'}])
    assert step.calls[0].arguments == {'query': 'Juniper'}
    final = await model.complete([{'role': 'tool', 'tool_call_id': 'call-1', 'content': '{}'}], [])
    assert final.content == 'Juniper uses Polaris.' and final.calls == ()
    assert 'tools' not in requests[1] and requests[1]['tool_choice'] == 'none'
    assert requests[0]['stream'] is False
    assert 'PRIVATE_REASONING' not in final.model_dump_json()


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('think', [None, False, True])
async def test_explicit_thinking_setting_reaches_provider_without_capturing_reasoning(structured, think):
    from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
    requests = []

    async def serve(request):
        requests.append(json.loads(request.content))
        content = 'Hello.'  # Both protocols use prose once tools are disabled.
        return httpx.Response(200, json=response({'content':content, 'reasoning_content':'PRIVATE_REASONING'}))

    provider = SelfHostedStructuredToolChat if structured else SelfHostedToolChat
    model = provider('http://127.0.0.1:11434/v1', 'test-model', think=think, transport=httpx.MockTransport(serve))
    step = await model.complete([{'role':'user','content':'Hello'}], [])
    assert step.content == 'Hello.'
    assert 'PRIVATE_REASONING' not in step.model_dump_json()
    if think is None:
        assert 'think' not in requests[0]
    else:
        assert requests[0]['think'] is think


@pytest.mark.parametrize('think', [0, 1, 'false', []])
def test_invalid_thinking_setting_fails_before_request(think):
    with pytest.raises(ValueError, match='thinking'):
        SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model', think=think)


@pytest.mark.parametrize('packet', [
    response({'content': 'partial'}, 'length'),
    response({'content': '', 'tool_calls': [tool()]}, 'stop'),
    response({'content': '', 'tool_calls': []}, 'tool_calls'),
    response({'tool_calls': [tool('{"query":"first","query":"second"}')]}, 'tool_calls'),
    response({'tool_calls': [tool('[]')]}, 'tool_calls'),
    response({'tool_calls': [tool('{"query":NaN}')]}, 'tool_calls'),
    response({'tool_calls': [tool('{"query":1e309}')]}, 'tool_calls'),
    response({'content': 'answer', 'tool_calls': ''}),
    response({'content': 'answer', 'function_call': {'name': 'search_memory'}}),
    response({'tool_calls': [tool(call_id='a'), tool(call_id='a')]}, 'tool_calls'),
    response({'tool_calls': [tool('x'*16001)]}, 'tool_calls'),
    response({'content': 'answer', 'refusal': 'refused'}),
])
async def test_malformed_or_incomplete_provider_output_fails_closed(packet):
    model = SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model',
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=packet)))
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model.complete([{'role': 'user', 'content': 'Hello'}], [{'type': 'function'}])


async def test_http_error_body_is_not_exposed():
    model = SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model',
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text='PRIVATE_KEY')))
    with pytest.raises(RuntimeError, match='tool model unavailable') as error:
        await model.complete([{'role': 'user', 'content': 'Hello'}], [])
    assert 'PRIVATE_KEY' not in str(error.value)


async def test_response_bytes_are_bounded():
    model = SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model', max_response_bytes=1024,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b'x'*1025)))
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model.complete([{'role': 'user', 'content': 'Hello'}], [])


@pytest.mark.parametrize('external', [False, True])
async def test_cleanup_cannot_suppress_deadline_or_external_cancellation(external):
    import asyncio
    entered = asyncio.Event()
    class SlowClose(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json=response({'content': 'late success'}))
        async def aclose(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
    model = SelfHostedToolChat('http://127.0.0.1:11434/v1', 'test-model',
                              timeout_s=1 if external else 0.01, transport=SlowClose())
    task = asyncio.create_task(model.complete([{'role': 'user', 'content': 'Hello'}], []))
    await entered.wait()
    if external:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if external else RuntimeError):
        await task
