from __future__ import annotations

import pytest
import httpx
import json

from .pipeline import Passage, messages, pack_context, rerank, unique_passages


@pytest.mark.asyncio
async def test_openrouter_uses_same_judgment_payload_and_keeps_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from .pipeline import rank
    monkeypatch.setenv('SCONE_CHAT_API_KEY', 'test-router-key')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'test-direct-key')
    calls: list[tuple[str, dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), body))
        router = request.url.host == 'openrouter.ai'
        assert request.headers['authorization'] == 'Bearer ' + ('test-router-key' if router else 'test-direct-key')
        return httpx.Response(200, json={'model': body['model'],
            'answers': {'p_0': {'type': 'noul', 'noul': .9}}, 'usage': {'input_tokens': 20, 'output_tokens': 5, 'cost': .001}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        passages = [Passage('a', 'A', 'The sky is blue.')]
        direct, _, _ = await rank(client, 'Which color?', passages)
        recovered, audit, _ = await rank(client, 'Which color?', passages, provider='openrouter')
    assert direct == recovered == {'a': .9}
    assert calls[1][0] == 'https://openrouter.ai/api/alpha/decisions'
    assert calls[1][1]['model'] == 'typesafe/jev-1.13-20260917'
    assert {k: v for k, v in calls[0][1].items() if k != 'model'} == {k: v for k, v in calls[1][1].items() if k != 'model'}
    assert audit['response'] == {'model': 'typesafe/jev-1.13-20260917',
        'answers': {'p_0': {'type': 'noul', 'noul': .9}}, 'usage': {'input_tokens': 20, 'output_tokens': 5, 'cost': .001}}


def test_byte_budget_preserves_unicode_and_limits_sources() -> None:
    passages = [Passage(str(i), str(i), 'é' * 100) for i in range(6)]
    context, ids = pack_context(passages, max_bytes=50)
    assert len(context.encode()) <= 50
    assert ids == ['0']
    assert '\ufffd' not in context
    assert len(pack_context(passages)[1]) == 5


def test_shared_scores_preserve_arm_tie_order() -> None:
    a, b = Passage('a', 'A', 'one'), Passage('b', 'B', 'two')
    assert unique_passages({'scone': [a, b], 'llamaindex': [b, a]}) == [a, b]
    assert rerank([b, a], {'a': .5, 'b': .5}) == [b, a]
    assert rerank([a, b], {'a': .1, 'b': .9}) == [b, a]
    with pytest.raises(ValueError, match='collision'):
        unique_passages({'scone': [a], 'llamaindex': [Passage('a', 'B', 'different')]})


def test_original_question_and_common_prompt() -> None:
    question = 'Which city?'
    result = messages(question, '[Source 1]\nLondon')
    assert result[-1] == {'role': 'user', 'content': question}
    assert 'INSUFFICIENT_EVIDENCE' in result[0]['content']
