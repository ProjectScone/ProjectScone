"""Tool inference failures are distinguishable without recording source text."""
import asyncio
import json
import logging

import httpx
import pytest

from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat


def packet(reason='stop', usage=None):
    return {'choices': [{'finish_reason': reason, 'message': {
        'role': 'assistant', 'content': 'PRIVATE_RESPONSE', 'reasoning_content': 'PRIVATE_REASONING'}}],
        'usage': usage}


def provider(transport, **kwargs):
    return SelfHostedToolChat('http://127.0.0.1:11434/v1', 'PRIVATE_MODEL',
        api_key='PRIVATE_KEY', transport=transport, **kwargs)


async def complete(model):
    return await model.complete([{'role': 'user', 'content': 'PRIVATE_QUESTION'}], [])


def finished(caplog):
    [row] = [row for row in caplog.records if getattr(row, 'event', None) == 'tool_model.finished']
    assert 'PRIVATE' not in caplog.text
    assert 'PRIVATE' not in repr(row.__dict__)
    return row


@pytest.fixture(autouse=True)
def capture(caplog):
    caplog.set_level(logging.INFO, logger='scone_memory.providers.tool_chat')


async def test_success_reports_usage_and_correlates_events_without_changing_request(caplog):
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=packet(usage={'prompt_tokens': 123, 'completion_tokens': 9, 'total_tokens': 132}))

    result = await complete(provider(httpx.MockTransport(serve)))
    assert result.content == 'PRIVATE_RESPONSE'
    row = finished(caplog)
    [start] = [record for record in caplog.records if getattr(record, 'event', None) == 'tool_model.started']
    assert start.call_id == row.call_id and row.call_id
    assert row.outcome == 'completed' and row.failure_kind is None
    assert row.protocol == 'native' and row.finish_reason == 'stop'
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (123, 9, 132)
    assert row.response_bytes > 0 and row.request_bytes > 0 and row.http_status == 200
    assert row.elapsed_ms >= row.headers_ms >= 0
    assert row.output_token_limit == 2048
    assert set(requests[0]) == {'model', 'messages', 'stream', 'temperature', 'max_tokens', 'tool_choice'}


@pytest.mark.parametrize('usage', [None, 'PRIVATE_USAGE', {'prompt_tokens': True, 'completion_tokens': -1, 'total_tokens': 'PRIVATE_USAGE'},
    {'prompt_tokens': 1.5, 'completion_tokens': 10**15, 'total_tokens': []}])
async def test_invalid_optional_usage_is_unknown_and_does_not_reject_valid_answer(caplog, usage):
    assert (await complete(provider(httpx.MockTransport(lambda _: httpx.Response(200, json=packet(usage=usage)))))).content == 'PRIVATE_RESPONSE'
    row = finished(caplog)
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (None, None, None)


@pytest.mark.parametrize('reason, expected', [('length', 'length'), ('content_filter', 'content_filter'), ('PRIVATE_REASON', 'unknown'), ([], 'unknown')])
async def test_incomplete_response_logs_safe_finish_reason_and_usage(caplog, reason, expected):
    model = provider(httpx.MockTransport(lambda _: httpx.Response(200, json=packet(reason, {'prompt_tokens': 8192}))))
    with pytest.raises(RuntimeError, match='^tool model unavailable$'):
        await complete(model)
    row = finished(caplog)
    assert row.outcome == 'failed' and row.failure_kind == 'invalid_response'
    assert row.phase == 'parse' and row.finish_reason == expected and row.prompt_tokens == 8192


async def test_http_failure_logs_status_without_reading_error_body(caplog):
    with pytest.raises(RuntimeError, match='^tool model unavailable$'):
        await complete(provider(httpx.MockTransport(lambda _: httpx.Response(503, text='PRIVATE_HTTP_ERROR'))))
    row = finished(caplog)
    assert row.failure_kind == 'http_error' and row.http_status == 503
    assert row.response_bytes == 0 and row.phase == 'response'


@pytest.mark.parametrize('error, expected', [(httpx.ReadTimeout, 'transport_timeout'), (httpx.ConnectError, 'transport_error')])
async def test_transport_failures_are_classified_without_exception_messages(caplog, error, expected):
    def fail(request):
        raise error('PRIVATE_EXCEPTION', request=request)

    with pytest.raises(RuntimeError, match='^tool model unavailable$'):
        await complete(provider(httpx.MockTransport(fail)))
    row = finished(caplog)
    assert row.failure_kind == expected and row.http_status is None


async def test_oversized_response_records_bytes_received_before_refusal(caplog):
    with pytest.raises(RuntimeError, match='^tool model unavailable$'):
        await complete(provider(httpx.MockTransport(lambda _: httpx.Response(200, content=b'x' * 1025)), max_response_bytes=1024))
    row = finished(caplog)
    assert row.failure_kind == 'response_bytes' and row.response_bytes == 1025
    assert row.phase == 'response' and row.finish_reason is None


async def test_partial_read_failure_keeps_byte_count_and_closes_response(caplog):
    closed = []

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'PRIVATE_PARTIAL_RESPONSE'
            raise httpx.ReadError('PRIVATE_READ_ERROR')

        async def aclose(self):
            closed.append(True)

    model = provider(httpx.MockTransport(lambda _: httpx.Response(200, stream=BrokenStream())))
    with pytest.raises(RuntimeError, match='^tool model unavailable$'):
        await complete(model)
    row = finished(caplog)
    assert row.response_bytes == len(b'PRIVATE_PARTIAL_RESPONSE')
    assert row.failure_kind == 'transport_error' and row.phase == 'response'
    assert row.finish_reason is None and closed == [True]


@pytest.mark.parametrize('external', [False, True])
async def test_request_deadline_and_external_cancellation_remain_distinct(caplog, external):
    entered = asyncio.Event()

    async def wait(request):
        entered.set()
        await asyncio.Event().wait()

    model = provider(httpx.MockTransport(wait), timeout_s=1 if external else 0.01)
    task = asyncio.create_task(complete(model))
    await entered.wait()
    if external:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if external else RuntimeError):
        await task
    row = finished(caplog)
    assert row.outcome == ('cancelled' if external else 'failed')
    assert row.failure_kind == (None if external else 'request_timeout')


@pytest.mark.parametrize('tools, protocol', [([], 'structured_answer'),
    ([{'type': 'function', 'function': {'name': 'search_memory'}}], 'structured_action')])
async def test_structured_phases_have_separate_protocol_labels(caplog, tools, protocol):
    def serve(request):
        response = packet()
        if tools:
            response['choices'][0]['message']['content'] = json.dumps({'action': 'answer', 'answer': 'PRIVATE_RESPONSE'})
        return httpx.Response(200, json=response)

    model = SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1', 'PRIVATE_MODEL', transport=httpx.MockTransport(serve))
    assert (await model.complete([{'role': 'user', 'content': 'PRIVATE_QUESTION'}], tools)).content == 'PRIVATE_RESPONSE'
    assert finished(caplog).protocol == protocol
