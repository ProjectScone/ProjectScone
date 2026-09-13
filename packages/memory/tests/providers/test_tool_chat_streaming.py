"""A tool turn that streams its public text as it is written.

The native tool turn is nonstreaming by design: one request, one bounded
body, one parse that owns acceptance. Answer streaming for agents needs
the same turn to hand over content deltas as they arrive -- and nothing
else: tool-call arguments accumulate silently, reasoning fields are never
read, and the final step goes through the very same parser as the
nonstreaming reply, so what a reader saw stream and what the loop accepts
are one reply.
"""

from __future__ import annotations

import json

import httpx
import pytest

from scone_memory.providers.tool_chat import SelfHostedToolChat

pytestmark = pytest.mark.asyncio

MESSAGES = [{'role': 'system', 'content': 'Answer briefly.'}, {'role': 'user', 'content': 'Say hello.'}]
TOOLS = [{'type': 'function', 'function': {'name': 'search_memory', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]


def chunk(delta: dict[str, object], finish: str | None = None, **extra: object) -> str:
    return 'data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}], **extra}) + '\n\n'


def served(body: str, requests: list[dict[str, object]], *, status: int = 200) -> httpx.MockTransport:
    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(status, content=body.encode(), headers={'content-type': 'text/event-stream'})
    return httpx.MockTransport(serve)


def chat(transport: httpx.MockTransport, **options: object) -> SelfHostedToolChat:
    return SelfHostedToolChat('http://127.0.0.1:11434/v1', 'local-model', transport=transport, **options)  # type: ignore[arg-type]


HELLO = (chunk({'role': 'assistant', 'content': ''}) + chunk({'content': 'Hel'}) + chunk({'content': 'lo'})
         + ': keep-alive\n\n' + chunk({}, 'stop') + 'data: [DONE]\n\n')


async def test_public_text_reaches_the_sink_in_order_and_the_step_is_the_joined_reply():
    requests: list[dict[str, object]] = []
    seen: list[str] = []

    async def sink(text: str) -> None:
        seen.append(text)

    step = await chat(served(HELLO, requests)).complete(MESSAGES, [], on_public_text=sink)
    assert seen == ['Hel', 'lo'], 'every content delta, in order, and nothing that was not content'
    assert step.content == 'Hello' and step.calls == ()
    assert requests[0]['stream'] is True and requests[0]['stream_options'] == {'include_usage': True}


async def test_a_tool_call_turn_streams_no_text_and_assembles_arguments_across_chunks():
    requests: list[dict[str, object]] = []
    seen: list[str] = []

    async def sink(text: str) -> None:
        seen.append(text)

    body = (chunk({'role': 'assistant', 'content': None})
            + chunk({'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function', 'function': {'name': 'search_memory', 'arguments': ''}}]})
            + chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '{"query":'}}]})
            + chunk({'tool_calls': [{'index': 0, 'function': {'arguments': ' "hello"}'}}]})
            + chunk({}, 'tool_calls') + 'data: [DONE]\n\n')
    step = await chat(served(body, requests)).complete(MESSAGES, TOOLS, on_public_text=sink)
    assert seen == [], 'arguments are not public text'
    assert [(call.id, call.name, call.arguments) for call in step.calls] == [('call-1', 'search_memory', {'query': 'hello'})]
    assert step.content == ''


async def test_usage_from_the_final_chunk_is_kept():
    body = HELLO.replace('data: [DONE]', 'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}) + '\n\ndata: [DONE]')

    async def sink(text: str) -> None:
        pass

    step = await chat(served(body, [])).complete(MESSAGES, [], on_public_text=sink)
    assert (step.usage.prompt_tokens, step.usage.completion_tokens, step.usage.total_tokens) == (3, 2, 5)


@pytest.mark.parametrize('body', [
    chunk({'content': 'Hello'}) + 'data: [DONE]\n\n',                                  # no finish
    chunk({'content': 'Hello'}) + chunk({}, 'stop') + chunk({'content': ' more'}) + 'data: [DONE]\n\n',  # text after finish
    chunk({'content': 'Hello'}),                                                        # cut off
    'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'content': 'a'}, 'finish_reason': None}, {'index': 1, 'delta': {}, 'finish_reason': None}]}) + '\n\n' + chunk({}, 'stop') + 'data: [DONE]\n\n',  # two choices
    chunk({'content': ''}) + chunk({}, 'stop') + 'data: [DONE]\n\n',                    # empty reply
    'data: not json\n\n' + chunk({}, 'stop') + 'data: [DONE]\n\n',                      # a line that is not a chunk
    chunk({'tool_calls': [{'index': 0, 'id': 'c', 'type': 'function', 'function': {'name': 'search_memory', 'arguments': '{}'}}]}) + chunk({}, 'stop') + 'data: [DONE]\n\n',  # calls but finish stop
    chunk({'refusal': 'I will not'}) + chunk({'content': 'but here'}) + chunk({}, 'stop') + 'data: [DONE]\n\n',  # a refusal is not a reply
])
async def test_a_stream_that_is_not_one_complete_reply_is_refused(body: str):
    async def sink(text: str) -> None:
        pass

    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await chat(served(body, [])).complete(MESSAGES, TOOLS, on_public_text=sink)


async def test_a_sink_that_fails_fails_the_turn_and_no_reply_is_returned():
    async def sink(text: str) -> None:
        raise RuntimeError('observation closed')

    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await chat(served(HELLO, [])).complete(MESSAGES, [], on_public_text=sink)


async def test_the_byte_budget_covers_the_whole_stream():
    body = ''.join(chunk({'content': 'x' * 200}) for _ in range(20)) + chunk({}, 'stop') + 'data: [DONE]\n\n'
    seen: list[str] = []

    async def sink(text: str) -> None:
        seen.append(text)

    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await chat(served(body, []), max_response_bytes=1024).complete(MESSAGES, [], on_public_text=sink)
    assert len(''.join(seen)) < 1024, 'the sink never receives more than the budget allows'


async def test_without_a_sink_the_request_is_the_nonstreaming_turn():
    requests: list[dict[str, object]] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Hello'}}]})

    step = await chat(httpx.MockTransport(serve)).complete(MESSAGES, [])
    assert step.content == 'Hello' and requests[0]['stream'] is False and 'stream_options' not in requests[0]


async def test_a_reasoning_field_in_a_delta_is_never_handed_to_the_sink():
    seen: list[str] = []

    async def sink(text: str) -> None:
        seen.append(text)

    body = chunk({'reasoning_content': 'let me think', 'content': 'Hi'}) + chunk({}, 'stop') + 'data: [DONE]\n\n'
    step = await chat(served(body, [])).complete(MESSAGES, [], on_public_text=sink)
    assert seen == ['Hi'] and step.content == 'Hi'


async def test_a_server_that_ignores_stream_still_answers_and_the_reader_gets_it_as_one_delta():
    seen: list[str] = []

    async def sink(text: str) -> None:
        seen.append(text)

    body = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Hello there'}}]})
    step = await chat(served(body, [])).complete(MESSAGES, [], on_public_text=sink)
    assert step.content == 'Hello there' and seen == ['Hello there']
