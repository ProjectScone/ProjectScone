from __future__ import annotations
import json
import httpx
import pytest
from scone_memory.providers.jev_sections import JevSectionChooser
from scone_memory.retrieval.section_routing import RouteMenu, RouteOption

MENU = RouteMenu(('Handbook',), (RouteOption('source:1', 'Cats', False),))
MODEL = 'typesafe/jev-1.13-20260917'


def response() -> dict[str, object]:
    return {'model': MODEL, 'answers': {'menu_0': {'type': 'choice', 'choice': 'option_0',
        'probabilities': {'option_0': .9, 'none': .1}, 'confidence': .8}},
        'usage': {'input_tokens': 10, 'output_tokens': 2}}


async def test_valid_addresses_are_mapped_without_generated_identifiers() -> None:
    calls = []
    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async with JevSectionChooser(api_key='test', transport=httpx.MockTransport(serve)) as chooser:
        result = await chooser.choose('Which animals purr?', (MENU,))
    assert result.distributions == ((.9, .1),)
    assert calls[0]['questions']['menu_0']['criteria']['option_0']['title'] == 'Cats'
    assert 'menu_0' in calls[0]['questions']['menu_0']['instructions']
    assert result.input_tokens == 10


@pytest.mark.parametrize('kind', ['foreign', 'nan', 'model', 'sum', 'choice'])
async def test_invalid_response_is_rejected(kind: str) -> None:
    raw = response()
    if kind == 'model':
        raw['model'] = 'other'
    else:
        answer = raw['answers']['menu_0']
        if kind == 'foreign':
            answer['probabilities'] = {'foreign': 1., 'none': 0.}
        elif kind == 'nan':
            answer['probabilities']['option_0'] = float('nan')
        elif kind == 'choice':
            answer['choice'] = 'unknown'
        else:
            answer['probabilities']['option_0'] = .3
    async with JevSectionChooser(api_key='test', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(raw)))) as chooser:
        with pytest.raises(ValueError):
            await chooser.choose('question', (MENU,))


async def test_failure_does_not_leak_provider_body() -> None:
    async with JevSectionChooser(api_key='test', transport=httpx.MockTransport(
            lambda request: httpx.Response(429, text='PRIVATE'))) as chooser:
        with pytest.raises(RuntimeError, match='unavailable') as error:
            await chooser.choose('question', (MENU,))
    assert 'PRIVATE' not in str(error.value)
