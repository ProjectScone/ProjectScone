import asyncio

import httpx
import pytest

from answer_target.probe import MODEL, generate


async def test_probe_records_complete_answer_and_full_request_time() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(.01)
        return httpx.Response(200, json={'model': MODEL, 'message': {'content': 'R7'},
            'done': True, 'done_reason': 'stop', 'eval_count': 2})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await generate(client, 'q', 'paper', 'code?', 'Code R7.')
    assert result.completed and result.answer == 'R7'
    assert result.wall_ms >= 10
    assert result.error is None


@pytest.mark.parametrize('done,reason', [(True, 'length'), (False, 'stop')])
async def test_probe_does_not_count_truncated_output_as_completed(done: bool, reason: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'model': MODEL, 'message': {'content': 'R7'},
            'done': done, 'done_reason': reason})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await generate(client, 'q', 'paper', 'code?', 'Code R7.')
    assert not result.completed and result.error == 'incomplete_generation'


async def test_probe_keeps_timeout_time_and_failed_row() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(.01)
        raise httpx.ReadTimeout('fixture')
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await generate(client, 'q', 'paper', 'code?', 'Code R7.')
    assert not result.completed and result.error == 'ReadTimeout'
    assert result.wall_ms >= 10 and result.id == 'q'


async def test_probe_rejects_a_different_model() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'model': 'unexpected-model', 'message': {'content': 'R7'},
            'done': True, 'done_reason': 'stop'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await generate(client, 'q', 'paper', 'code?', 'Code R7.')
    assert not result.completed and result.error == 'ValueError'
